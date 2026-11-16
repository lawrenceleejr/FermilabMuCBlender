#!/usr/bin/env python3
"""Reduce the metro cache (tools/fetch_metro.py) to the rasters and vectors the
metro view renders from, in assets/metro/.

    python3 tools/bake_metro.py

Every raster is resampled onto the scene's own plan grid -- the orthographic
tangent plane at Wilson Hall, see scene/metro_geo.py -- so the builder never
touches latitude, Mercator tiles or projections: texel (i, j) of a region is
plan position (x0 + (j + 0.5) * res, y1 - (i + 0.5) * res).

Outputs, per region (near / far):

  dem_<r>.png       16-bit elevation, metres = lo + v / 65535 * (hi - lo)
  aux_<r>.png       R: cartographic relief shade (0.5 = flat), G: water mask
  albedo_<r>.jpg    USGS orthoimagery, sRGB
  lights_<r>.png    R: sodium lamps, G: LED lamps, B: diffuse urban glow;
                    lamp density = dmax * (v / 255) ** 2.2  [lamps per m^2]

and metro_plan.json: the regions, raster encodings, the Chicago city limits, the
UChicago campus, the Loop, attribution.

The lights. A city at night is its street network, lit. So the lamps are placed
on the real streets -- TIGER/Line, every road in the eleven counties under the
near region, at a spacing by road class -- and then *gated by what VIIRS
actually sees from orbit*: NASA's Black Marble says where the light is, TIGER
says where the streets are. Without the gate every farm road in Kane County
would be lit like a suburb; with it the farmland goes dark the way it really
does. Beyond the near region, where one pixel is hundreds of metres of ground,
lamps are drawn at the density Black Marble implies, calibrated against the near
region's own streets, plus the primary and secondary roads of four states.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from scene import metro_geo as G  # noqa: E402

CACHE = os.path.join(ROOT, "assets", "metro", "cache")
OUT = os.path.join(ROOT, "assets", "metro")
Image.MAX_IMAGE_PIXELS = None

# Same vertical datum as the site plan, so the two terrains meet
SITE_PLAN = os.path.join(ROOT, "assets", "plan", "site_plan.json")
LAKE_MICHIGAN_M = 176.6      # mean surface, IGLD85 ~ 176.5 m; the DEM reads 175-176 on it

# Lamp spacing (m) and relative output by TIGER MTFCC road class
ROAD_CLASS = {
    "S1100": (40.0, 1.8),    # primary: interstates, high-mast lit through the metro
    "S1200": (40.0, 1.4),    # secondary: US/state routes, arterials
    "S1630": (35.0, 1.4),    # ramps
    "S1400": (45.0, 1.0),    # local streets
    "S1640": (50.0, 0.6),    # service drives
    "S1730": (55.0, 0.45),   # alleys -- Chicago's are lit
    "S1780": (30.0, 0.9),    # parking-lot roads: a lit retail lot reads as a dense patch
}
LOCAL = {"S1400", "S1640", "S1730", "S1780"}
# Black Marble gate (tile value 0..255): below LO a road is dark, above HI fully lit
GATE_LOCAL = (25.0, 110.0)
GATE_TRUNK = (14.0, 70.0)    # trunk roads are lit further into the exurbs
DMAX = {"near": 0.02, "far": 0.0035}    # lamps per m^2 at code 255


# --------------------------------------------------------------------------- #
# tiles
# --------------------------------------------------------------------------- #
def mosaic(name, mode):
    meta = json.load(open(os.path.join(CACHE, name, "tiles.json")))
    z, x0, x1, y0, y1, ext = meta["z"], meta["x0"], meta["x1"], meta["y0"], meta["y1"], meta["ext"]
    d = os.path.join(CACHE, name, str(z))
    first = None
    rows = []
    for y in range(y0, y1 + 1):
        row = []
        for x in range(x0, x1 + 1):
            p = os.path.join(d, f"{x}_{y}.{ext}")
            try:
                im = np.asarray(Image.open(p).convert(mode))
            except Exception:  # noqa: BLE001
                im = None
            if im is not None and first is None:
                first = im
            row.append(im)
        rows.append(row)
    blank = np.zeros_like(first)
    m = np.vstack([np.hstack([t if t is not None else blank for t in r]) for r in rows])
    ts = first.shape[0]
    return m, dict(z=z, x0=x0, y0=y0, ts=ts)


def sample(m, g, lat, lon, *, nearest=False):
    """Bilinear sample of a Web-Mercator mosaic at lat/lon arrays."""
    n = 2 ** g["z"]
    fx = ((lon + 180.0) / 360.0 * n - g["x0"]) * g["ts"] - 0.5
    fy = ((1.0 - np.arcsinh(np.tan(np.radians(lat))) / math.pi) / 2.0 * n - g["y0"]) * g["ts"] - 0.5
    H, W = m.shape[:2]
    fx = np.clip(fx, 0, W - 1.001)
    fy = np.clip(fy, 0, H - 1.001)
    if nearest:
        return m[np.round(fy).astype(int), np.round(fx).astype(int)]
    i0, j0 = fy.astype(int), fx.astype(int)
    u, v = fx - j0, fy - i0
    if m.ndim == 3:
        u, v = u[..., None], v[..., None]
    mf = m.astype(np.float32) if m.dtype != np.float32 else m
    a = mf[i0, j0] * (1 - u) + mf[i0, j0 + 1] * u
    b = mf[i0 + 1, j0] * (1 - u) + mf[i0 + 1, j0 + 1] * u
    return a * (1 - v) + b * v


def plan_grid(region, res, r0=None, r1=None):
    x0, x1, y0, y1 = region
    rows, cols = G.grid_shape(region, res)
    r0 = 0 if r0 is None else r0
    r1 = rows if r1 is None else r1
    xs = x0 + (np.arange(cols) + 0.5) * res
    ys = y1 - (np.arange(r0, r1) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    return X, Y


def resample(m, g, region, res, chunk=600, **kw):
    rows, cols = G.grid_shape(region, res)
    out = None
    for r0 in range(0, rows, chunk):
        r1 = min(rows, r0 + chunk)
        X, Y = plan_grid(region, res, r0, r1)
        lat, lon = G.np_to_latlon(X, Y)
        s = sample(m, g, lat, lon, **kw)
        if out is None:
            out = np.zeros((rows, cols) + s.shape[2:], np.float32)
        out[r0:r1] = s
    return out


def terrarium(m):
    m = m.astype(np.float32)
    return m[..., 0] * 256.0 + m[..., 1] + m[..., 2] / 256.0 - 32768.0


# --------------------------------------------------------------------------- #
def bake_dem(tag, region, res, datum):
    m, g = mosaic(f"dem_{tag}", "RGB")
    E = terrarium(m)
    # the terrarium set has a few no-data pits and spikes at this scale; clamp to
    # the physical range of the region (lake bed excluded: it is drawn as water)
    E = np.clip(E, 150.0, 420.0)
    dem = resample(E, g, region, res)
    lo, hi = float(dem.min()), float(dem.max())
    q = np.round((dem - lo) / (hi - lo) * 65535.0).astype(np.uint16)
    Image.fromarray(q).save(os.path.join(OUT, f"dem_{tag}.png"))

    # water: at or below the lake surface, and connected to the open lake
    from scipy import ndimage
    wet = dem <= LAKE_MICHIGAN_M
    lab, n = ndimage.label(wet)
    if n:
        sizes = ndimage.sum(wet, lab, range(1, n + 1))
        keep = np.zeros(n + 1, bool)
        keep[1:] = sizes >= max(sizes.max() * 0.02, 50)
        wet = keep[lab]
    wet = ndimage.binary_closing(wet, iterations=2)

    # cartographic relief: classic NW light at 45 deg, 4x vertical exaggeration.
    # Northern Illinois is a glacial plain -- 267 m of relief over 60 km -- and
    # at twilight there is no sun to rake it, so without this the topography is
    # simply not visible from 28 km. It is a map convention, applied gently in
    # the shader (--relief), not a claim about the light.
    gy, gx = np.gradient(dem * 4.0, res)
    az, alt = math.radians(315.0), math.radians(45.0)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    flat = math.sin(alt)
    rel = np.clip(0.5 + (shade - flat) * 1.2, 0.0, 1.0)
    rel[wet] = 0.5
    aux = np.zeros(dem.shape + (3,), np.uint8)
    aux[..., 0] = np.round(rel * 255).astype(np.uint8)
    aux[..., 1] = wet.astype(np.uint8) * 255
    Image.fromarray(aux).save(os.path.join(OUT, f"aux_{tag}.png"))
    print(f"[bake] dem_{tag}: {dem.shape[1]}x{dem.shape[0]} @ {res:.0f} m, "
          f"{lo:.0f}..{hi:.0f} m, water {wet.mean() * 100:.1f} %")
    return dict(file=f"dem_{tag}.png", aux=f"aux_{tag}.png", res=res, lo=lo, hi=hi,
                datum=datum, shape=list(dem.shape)), dem, wet


def bake_albedo(tag, region, res):
    m, g = mosaic(f"imagery_{tag}", "RGB")
    a = resample(m, g, region, res)
    Image.fromarray(np.clip(np.round(a), 0, 255).astype(np.uint8)).save(
        os.path.join(OUT, f"albedo_{tag}.jpg"), quality=90, subsampling=0)
    print(f"[bake] albedo_{tag}: {a.shape[1]}x{a.shape[0]} @ {res:.0f} m")
    return dict(file=f"albedo_{tag}.jpg", res=res)


# --------------------------------------------------------------------------- #
# lights
# --------------------------------------------------------------------------- #
def read_lines(path, classes):
    import shapefile
    r = shapefile.Reader(path)
    out = []
    for sr in r.iterShapeRecords():
        c = sr.record["MTFCC"]
        if c not in classes:
            continue
        pts = sr.shape.points
        parts = list(sr.shape.parts) + [len(pts)]
        for a, b in zip(parts[:-1], parts[1:]):
            if b - a >= 2:
                out.append((c, np.asarray(pts[a:b], np.float64)))
    return out


def resample_line(xy, spacing, rng):
    seg = np.diff(xy, axis=0)
    L = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(L)])
    tot = cum[-1]
    if tot < spacing * 0.5:
        return xy[:1]
    s = np.arange(rng.uniform(0, spacing), tot, spacing)
    k = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(L) - 1)
    f = ((s - cum[k]) / np.maximum(L[k], 1e-9))[:, None]
    return xy[k] + seg[k] * f


def splat(raster, region, res, x, y, w):
    x0, x1, y0, y1 = region
    fx = (x - x0) / res - 0.5
    fy = (y1 - y) / res - 0.5
    H, W = raster.shape
    ok = (fx >= 0) & (fy >= 0) & (fx < W - 1) & (fy < H - 1)
    fx, fy, w = fx[ok], fy[ok], w[ok]
    j, i = fx.astype(int), fy.astype(int)
    u, v = fx - j, fy - i
    np.add.at(raster, (i, j), w * (1 - u) * (1 - v))
    np.add.at(raster, (i, j + 1), w * u * (1 - v))
    np.add.at(raster, (i + 1, j), w * (1 - u) * v)
    np.add.at(raster, (i + 1, j + 1), w * u * v)


def gate(v, lohi):
    lo, hi = lohi
    t = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def in_poly_mask(region, res, polys):
    """Rasterise plan-space polygons to a boolean mask on a region grid."""
    from PIL import ImageDraw
    x0, x1, y0, y1 = region
    rows, cols = G.grid_shape(region, res)
    im = Image.new("L", (cols, rows), 0)
    dr = ImageDraw.Draw(im)
    for k, poly in enumerate(polys):
        pts = [((x - x0) / res, (y1 - y) / res) for x, y in poly]
        dr.polygon(pts, fill=255 if k == 0 else 0)     # first = outer, rest = holes
    return np.asarray(im) > 0


def encode(dens, dmax):
    return np.round(np.clip(dens / dmax, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


def bake_lights(near_wet, far_wet, chicago_polys):
    t0 = time.time()
    bm_m, bm_g = mosaic("nightlights", "L")
    bm = bm_m.astype(np.float32)
    rng = np.random.default_rng(7)
    NEAR, FAR = G.NEAR, G.FAR
    rn, rf = G.RES["lights_near"], G.RES["lights_far"]
    shp_n, shp_f = G.grid_shape(NEAR, rn), G.grid_shape(FAR, rf)
    sod_n, led_n = np.zeros(shp_n, np.float32), np.zeros(shp_n, np.float32)
    sod_f, led_f = np.zeros(shp_f, np.float32), np.zeros(shp_f, np.float32)
    local_f = np.zeros(shp_f, np.float32)        # local streets only, for calibration

    chi_mask_n = in_poly_mask(NEAR, rn, chicago_polys)
    chi_mask_f = in_poly_mask(FAR, rf, chicago_polys)

    def is_chicago(x, y, mask, region, res):
        x0, x1, y0, y1 = region
        j = np.clip(((x - x0) / res).astype(int), 0, mask.shape[1] - 1)
        i = np.clip(((y1 - y) / res).astype(int), 0, mask.shape[0] - 1)
        return mask[i, j]

    def place(lines, *, near):
        n_lamps = 0
        by_class = {}
        for c, ll in lines:
            by_class.setdefault(c, []).append(ll)
        for c, group in by_class.items():
            spacing, wt = ROAD_CLASS[c]
            # resample in metres: project each line first
            xs, ys = [], []
            for ll in group:
                x, y = G.np_to_xy(ll[:, 1], ll[:, 0])
                p = resample_line(np.stack([x, y], 1), spacing, rng)
                xs.append(p[:, 0])
                ys.append(p[:, 1])
            if not xs:
                continue
            x = np.concatenate(xs)
            y = np.concatenate(ys)
            lat, lon = G.np_to_latlon(x, y)
            b = sample(bm, bm_g, lat, lon)
            w = wt * gate(b, GATE_LOCAL if c in LOCAL else GATE_TRUNK)
            # small per-lamp spread so a street is not a row of identical dots
            w = w * rng.uniform(0.55, 1.45, size=w.shape)
            keep = w > 0.02
            x, y, w = x[keep], y[keep], w[keep]
            chi = is_chicago(x, y, chi_mask_f, FAR, rf)
            if near:
                splat(sod_n, NEAR, rn, x[~chi], y[~chi], w[~chi])
                splat(led_n, NEAR, rn, x[chi], y[chi], w[chi])
            else:
                # trunk roads beyond the near region only: inside it TIGER all-roads has them
                inn = (x > NEAR[0]) & (x < NEAR[1]) & (y > NEAR[2]) & (y < NEAR[3])
                x, y, w, chi = x[~inn], y[~inn], w[~inn], chi[~inn]
            splat(sod_f, FAR, rf, x[~chi], y[~chi], w[~chi])
            splat(led_f, FAR, rf, x[chi], y[chi], w[chi])
            if near and c in LOCAL:
                splat(local_f, FAR, rf, x, y, w)
            n_lamps += len(x)
            print(f"[bake]   {c}: {len(x)} lamps")
        return n_lamps

    n_near = 0
    for p in sorted(glob.glob(os.path.join(CACHE, "tiger", "tl_2024_*_roads.zip"))):
        print(f"[bake] lamps from {os.path.basename(p)}")
        n_near += place(read_lines(p, set(ROAD_CLASS)), near=True)
    n_trunk = 0
    for p in sorted(glob.glob(os.path.join(CACHE, "tiger", "tl_2024_*_prisecroads.zip"))):
        print(f"[bake] trunk lamps from {os.path.basename(p)}")
        n_trunk += place(read_lines(p, {"S1100", "S1200"}), near=False)

    # --- Black Marble on the far grid ------------------------------------------
    X, Y = plan_grid(FAR, rf)
    lat, lon = G.np_to_latlon(X, Y)
    bm_f = sample(bm, bm_g, lat, lon)
    del X, Y, lat, lon
    near_cells = np.zeros(shp_f, bool)
    x0, x1, y0, y1 = FAR
    j0, j1 = int((NEAR[0] - x0) / rf), int((NEAR[1] - x0) / rf)
    i0, i1 = int((y1 - NEAR[3]) / rf), int((y1 - NEAR[2]) / rf)
    near_cells[i0:i1, j0:j1] = True
    # calibrate: local-street lamp weight per m^2 as a function of Black Marble,
    # measured where both exist (inside NEAR, on land)
    land = ~far_wet
    sel = near_cells & land
    bins = np.clip(bm_f[sel].astype(int), 0, 255)
    dens = local_f[sel] / (rf * rf)
    tot = np.bincount(bins, weights=dens, minlength=256)
    cnt = np.bincount(bins, minlength=256)
    f = np.where(cnt > 30, tot / np.maximum(cnt, 1), np.nan)
    # fill gaps monotonically from the measured bins
    idx = np.arange(256)
    good = ~np.isnan(f)
    f = np.interp(idx, idx[good], f[good])
    f = np.maximum.accumulate(f)                # brighter from orbit -> not fewer lamps
    print("[bake] calibration (Black Marble -> local lamps/ha): " +
          ", ".join(f"{v}:{f[v] * 1e4:.1f}" for v in (20, 40, 80, 120, 180, 255)))
    lam = f[np.clip(bm_f.astype(int), 0, 255)] * rf * rf      # expected local lamp weight per texel
    mean_w = 0.9
    synth = rng.poisson(lam / mean_w).astype(np.float32) * mean_w
    synth[near_cells | far_wet] = 0.0
    sod_f += synth
    print(f"[bake] synthesised {synth.sum() / mean_w:.0f} lamps beyond the near region")

    # diffuse urban glow (lit lots, facades, sky-glow on the ground), from VIIRS
    from scipy import ndimage
    glow_f = np.clip((bm_f - 18.0) / 237.0, 0, 1) ** 1.6
    glow_f[far_wet] = 0
    glow_f = ndimage.gaussian_filter(glow_f, 1.0)
    Xn, Yn = plan_grid(NEAR, rn)
    latn, lonn = G.np_to_latlon(Xn, Yn)
    del Xn, Yn
    glow_n = np.clip((sample(bm, bm_g, latn, lonn) - 18.0) / 237.0, 0, 1) ** 1.6
    del latn, lonn
    near_wet_up = np.asarray(Image.fromarray(near_wet.astype(np.uint8) * 255).resize(
        (shp_n[1], shp_n[0]), Image.NEAREST)) > 0
    glow_n[near_wet_up] = 0

    # lamps never stand in the lake
    sod_n[near_wet_up] = led_n[near_wet_up] = 0
    sod_f[far_wet] = led_f[far_wet] = 0

    for tag, sod, led, glow, res in (("near", sod_n, led_n, glow_n, rn), ("far", sod_f, led_f, glow_f, rf)):
        area = res * res
        rgb = np.stack([encode(sod / area, DMAX[tag]), encode(led / area, DMAX[tag]),
                        np.round(np.clip(glow, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)], -1)
        Image.fromarray(rgb).save(os.path.join(OUT, f"lights_{tag}.png"), optimize=False, compress_level=6)
        sat = float(((sod + led) / area > DMAX[tag]).mean() * 100)
        print(f"[bake] lights_{tag}: {rgb.shape[1]}x{rgb.shape[0]} @ {res:.0f} m, {sat:.2f} % saturated")
    print(f"[bake] lights: {n_near} street lamps (near), {n_trunk} trunk lamps (far), "
          f"{time.time() - t0:.0f}s")
    return dict(near=dict(file="lights_near.png", res=rn, dmax=DMAX["near"]),
                far=dict(file="lights_far.png", res=rf, dmax=DMAX["far"]),
                encoding="density = dmax * (v/255)^2.2 lamps per m^2; R sodium, G LED, B glow (v/255)^2.2",
                lamps_near=n_near, lamps_trunk=n_trunk)


# --------------------------------------------------------------------------- #
# vectors
# --------------------------------------------------------------------------- #
def rdp(pts, tol):
    if len(pts) < 3:
        return pts
    a, b = np.asarray(pts[0]), np.asarray(pts[-1])
    ab = b - a
    nrm = np.hypot(*ab) or 1e-9
    q = np.asarray(pts) - a
    d = np.abs(ab[0] * q[:, 1] - ab[1] * q[:, 0]) / nrm
    k = int(np.argmax(d))
    if d[k] > tol:
        return rdp(pts[:k + 1], tol)[:-1] + rdp(pts[k:], tol)
    return [pts[0], pts[-1]]


def rdp_ring(pts, tol):
    """Simplify a closed ring: split at the vertex farthest from the first, so
    neither half has coincident endpoints (which would make every offset zero)."""
    pts = [tuple(p) for p in pts]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 4:
        return pts
    a = np.asarray(pts[0])
    k = int(np.argmax(np.hypot(*(np.asarray(pts) - a).T)))
    return rdp(pts[:k + 1], tol)[:-1] + rdp(pts[k:] + [pts[0]], tol)[:-1]


def chicago_rings():
    d = json.load(open(os.path.join(CACHE, "chicago_r122604.geojson")))
    g = d["features"][0]["geometry"]
    polys = g["coordinates"] if g["type"] == "Polygon" else [r for p in g["coordinates"] for r in p]
    out = []
    for ring in polys:
        ll = np.asarray(ring)
        x, y = G.np_to_xy(ll[:, 1], ll[:, 0])
        r = [[round(float(a), 1), round(float(b), 1)] for a, b in rdp_ring(list(zip(x, y)), 8.0)]
        if len(r) >= 4:                          # tiny enclaves collapse under 8 m simplification
            out.append(r)
    return out


def uchicago_polys():
    from shapely.geometry import Polygon, Point
    from shapely.ops import unary_union
    d = json.load(open(os.path.join(CACHE, "uchicago_landuse.json")))
    ps = []
    for poly in d["polygons"]:
        ll = np.asarray(poly)
        x, y = G.np_to_xy(ll[:, 0], ll[:, 1])
        p = Polygon(list(zip(x, y))).buffer(0)
        if p.area > 0:
            ps.append(p)
    u = unary_union(ps).buffer(8.0).buffer(-8.0)       # weld the tile seams
    quad = Point(*G.to_xy(*G.UCHICAGO_QUAD))
    parts = list(u.geoms) if hasattr(u, "geoms") else [u]
    # the campus is the part containing the quad, plus anything touching it
    # within 150 m (the medical campus, the Midway's south side)
    main = [p for p in parts if p.distance(quad) < 1.0]
    if not main:
        main = [min(parts, key=lambda p: p.distance(quad))]
    keep = [p for p in parts if min(p.distance(m) for m in main) < 150.0]
    out = []
    for p in keep:
        xy = list(p.exterior.coords)
        out.append([[round(a, 1), round(b, 1)] for a, b in rdp_ring(xy, 4.0)])
    area = sum(p.area for p in keep) / 1e6
    print(f"[bake] UChicago campus: {len(keep)} parts, {area:.2f} km^2")
    return out, area


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    datum = json.load(open(SITE_PLAN))["dem"]["elev_at_origin_m"]
    dem_n, _, wet_n = bake_dem("near", G.NEAR, G.RES["dem_near"], datum)
    dem_f, _, wet_f = bake_dem("far", G.FAR, G.RES["dem_far"], datum)
    alb_n = bake_albedo("near", G.NEAR, G.RES["albedo_near"])
    alb_f = bake_albedo("far", G.FAR, G.RES["albedo_far"])
    chi = chicago_rings()
    uc, uc_area = uchicago_polys()
    # far water mask on the lights grid
    wf = np.asarray(Image.fromarray(wet_f.astype(np.uint8) * 255).resize(
        G.grid_shape(G.FAR, G.RES["lights_far"])[::-1], Image.NEAREST)) > 0
    lights = bake_lights(wet_n, wf, chi)

    from shapely.geometry import Polygon
    chi_area = (Polygon(chi[0]).area - sum(Polygon(h).area for h in chi[1:])) / 1e6
    plan = {
        "origin": {"lat": G.LAT0, "lon": G.LON0,
                   "note": "orthographic tangent plane at OSM Wilson Hall; x east, y north, z up, metres"},
        "regions": {"near": list(G.NEAR), "far": list(G.FAR)},
        "dem": {"near": dem_n, "far": dem_f},
        "albedo": {"near": alb_n, "far": alb_f},
        "lights": lights,
        "lake_surface_m": LAKE_MICHIGAN_M,
        "chicago_limits": chi,
        "chicago_area_km2": round(chi_area, 1),
        "uchicago_campus": uc,
        "uchicago_area_km2": round(uc_area, 2),
        "willis_tower": list(G.to_xy(*G.WILLIS_TOWER)),
        "attribution": [
            "Elevation: AWS Terrain Tiles (Mapzen terrarium; SRTM, NED, ETOPO1)",
            "Imagery: USGS The National Map, USGSImageryOnly (public domain)",
            "Night lights: NASA Earth Observatory Black Marble 2016, VIIRS DNB (public domain)",
            "Streets: US Census Bureau TIGER/Line 2024 (public domain)",
            "City limits, campus: (c) OpenStreetMap contributors, ODbL",
        ],
    }
    with open(os.path.join(OUT, "metro_plan.json"), "w") as fh:
        json.dump(plan, fh)
    print(f"[bake] Chicago {chi_area:.0f} km^2 in {len(chi)} rings; wrote metro_plan.json "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
