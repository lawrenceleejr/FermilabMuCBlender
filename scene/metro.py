"""The metro view: Fermilab on the left, the City of Chicago on the right, from 28 km.

Built from assets/metro/ (tools/fetch_metro.py + tools/bake_metro.py):

* **The ground is the curved Earth.** One mesh on the orthographic tangent plane
  at Wilson Hall (see scene/metro_geo.py), each vertex at its true 3D position:
  real DEM height plus the sphere's drop, 2.4 km at the top of the frame. It is
  dense (150 m) where a pixel is tens of metres of ground and coarse (1 km)
  where foreshortening makes a pixel hundreds of metres deep.
* **The ground's colour is the map.** USGS orthoimagery as albedo, lit by the
  same twilight sky as the site views, with a gentle cartographic relief shade
  so the glacial topography -- moraines, the Fox and Des Plaines valleys --
  reads at a scale where twilight alone would flatten it.
* **The lights are the streets.** An emission raster of lamps on the TIGER
  street network, gated by NASA Black Marble, sodium in the suburbs and LED
  inside the city limits (Chicago replaced ~270 000 sodium heads with LEDs from
  2017). Two registers -- 15 m texels in the near half of the frame, 60 m beyond
  -- that hold the same lamps per square metre, cross-faded over 4 km.
* **Outlines** in the site figure's own vocabulary: boundaries are the warm
  white of the campus boundary, the proposed chain is cyan, the existing
  machines amber. A comparison ring (LHC, FCC, SSC, VLHC) is violet.

Everything the site views use for light -- the twilight world, exposure,
compositor -- is shared unchanged, so this frame and the site frames are lit by
the same sky at the same instant.
"""
from __future__ import annotations

import json
import math
import os

import bpy
import numpy as np

from . import common as C
from . import metro_geo as G
from . import site

DIR = os.path.join(C.ROOT, "assets", "metro")
PLAN = os.path.join(DIR, "metro_plan.json")

BOUNDARY_WHITE = (1.0, 0.93, 0.82)          # site.build_campus_boundary's colour
EXISTING_AMBER = (1.0, 0.62, 0.22)
COMPARE_VIOLET = (0.62, 0.36, 1.0)       # bright emitters desaturate under AgX: keep it dim and saturated

_plan = None
_grid = None            # (xs, ys, Z) of the terrain mesh, for draping overlays


def plan():
    global _plan
    if _plan is None:
        if not os.path.exists(PLAN):
            raise FileNotFoundError(f"{PLAN} missing: run tools/fetch_metro.py then tools/bake_metro.py")
        with open(PLAN) as fh:
            _plan = json.load(fh)
    return _plan


def available():
    return os.path.exists(PLAN)


# --------------------------------------------------------------------------- #
# rasters
# --------------------------------------------------------------------------- #
def _load(name, *, non_color=True):
    path = os.path.join(DIR, name)
    img = bpy.data.images.load(path, check_existing=True)
    img.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
    return img


def _dem_array(tag):
    d = plan()["dem"][tag]
    img = _load(d["file"])
    w, h = img.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    a = buf.reshape(h, w, 4)[::-1, :, 0]                 # top row = north
    return a * (d["hi"] - d["lo"]) + d["lo"], d


def _sample(arr, region, res, x, y):
    x0, x1, y0, y1 = region
    fx = np.clip((x - x0) / res - 0.5, 0, arr.shape[1] - 1.001)
    fy = np.clip((y1 - y) / res - 0.5, 0, arr.shape[0] - 1.001)
    j, i = fx.astype(int), fy.astype(int)
    u, v = fx - j, fy - i
    a = arr[i, j] * (1 - u) + arr[i, j + 1] * u
    b = arr[i + 1, j] * (1 - u) + arr[i + 1, j + 1] * u
    return a * (1 - v) + b * v


def _axis(lo, hi, nlo, nhi, fine, coarse):
    """Grid positions: `fine` spacing over [nlo, nhi], `coarse` outside."""
    out = list(np.arange(nlo, nhi + 1e-6, fine))
    x = nlo
    while x - coarse > lo:
        x -= coarse
        out.insert(0, x)
    out.insert(0, lo)
    x = nhi
    while x + coarse < hi:
        x += coarse
        out.append(x)
    out.append(hi)
    return np.unique(np.asarray(out, np.float64))


# --------------------------------------------------------------------------- #
# terrain
# --------------------------------------------------------------------------- #
def build_terrain(col, *, fine=150.0, coarse=1000.0, relief=0.55, albedo_gain=0.85,
                  saturation=0.8, lamp_strength=60.0, glow_strength=0.015, feather=4000.0):
    global _grid
    p = plan()
    near, far = p["regions"]["near"], p["regions"]["far"]
    xs = _axis(far[0], far[1], near[0], near[1], fine, coarse)
    ys = _axis(far[2], far[3], near[2], near[3], fine, coarse)
    X, Y = np.meshgrid(xs, ys)
    dn, mn = _dem_array("near")
    df, mf = _dem_array("far")
    datum = mn["datum"]
    zn = _sample(dn, near, mn["res"], X, Y)
    zf = _sample(df, far, mf["res"], X, Y)
    inside = (X >= near[0]) & (X <= near[1]) & (Y >= near[2]) & (Y <= near[3])
    elev = np.where(inside, zn, zf)
    lake = p["lake_surface_m"]
    elev = np.maximum(elev, lake)                     # the lake is a flat surface, not its bed
    drop = np.sqrt(np.maximum(G.R_EARTH ** 2 - X ** 2 - Y ** 2, 0.0)) - G.R_EARTH
    Z = elev - datum + drop
    _grid = (xs, ys, Z)

    ny, nx = Z.shape
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1).astype(np.float32)
    ii, jj = np.meshgrid(np.arange(ny - 1), np.arange(nx - 1), indexing="ij")
    a = (ii * nx + jj).ravel()
    faces = np.stack([a, a + 1, a + nx + 1, a + nx], 1)
    me = bpy.data.meshes.new("metro_terrain")
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", verts.ravel())
    me.loops.add(faces.size)
    me.loops.foreach_set("vertex_index", faces.ravel().astype(np.int32))
    me.polygons.add(len(faces))
    me.polygons.foreach_set("loop_start", (np.arange(len(faces)) * 4).astype(np.int32))
    me.update(calc_edges=True)
    me.validate(verbose=False)
    me.polygons.foreach_set("use_smooth", np.ones(len(faces), bool))
    obj = bpy.data.objects.new("metro_terrain", me)
    C.link_object(obj, col)
    obj.data.materials.append(terrain_material(relief=relief, albedo_gain=albedo_gain,
                                               saturation=saturation, lamp_strength=lamp_strength,
                                               glow_strength=glow_strength, feather=feather))
    print(f"[metro] terrain {nx}x{ny} = {len(verts)} verts over "
          f"{(far[1] - far[0]) / 1000:.0f} x {(far[3] - far[2]) / 1000:.0f} km, "
          f"elev {elev.min():.0f}..{elev.max():.0f} m, Earth drop to {drop.min():.0f} m")
    return obj


def height(x, y):
    """Ground z of the terrain mesh at plan (x, y): bilinear on its own grid."""
    xs, ys, Z = _grid
    i = np.clip(np.searchsorted(ys, y) - 1, 0, len(ys) - 2)
    j = np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2)
    u = np.clip((x - xs[j]) / (xs[j + 1] - xs[j]), 0, 1)
    v = np.clip((y - ys[i]) / (ys[i + 1] - ys[i]), 0, 1)
    return (Z[i, j] * (1 - u) * (1 - v) + Z[i, j + 1] * u * (1 - v)
            + Z[i + 1, j] * (1 - u) * v + Z[i + 1, j + 1] * u * v)


def terrain_material(*, relief, albedo_gain, saturation, lamp_strength, glow_strength, feather):
    p = plan()
    near, far = p["regions"]["near"], p["regions"]["far"]
    L = p["lights"]
    mat, nt, bsdf, _ = C.new_material("metro_ground")
    mat.cycles.emission_sampling = "NONE"     # the lamps light nothing at 28 km; keep them out of the light tree
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])
    X, Y = sep.outputs["X"], sep.outputs["Y"]

    def uv(region):
        x0, x1, y0, y1 = region
        u = C.nmath(nt, "MULTIPLY_ADD", X, value_b=1.0 / (x1 - x0))
        u.inputs[2].default_value = -x0 / (x1 - x0)
        v = C.nmath(nt, "MULTIPLY_ADD", Y, value_b=1.0 / (y1 - y0))
        v.inputs[2].default_value = -y0 / (y1 - y0)
        cv = nt.nodes.new("ShaderNodeCombineXYZ")
        nt.links.new(u.outputs[0], cv.inputs["X"])
        nt.links.new(v.outputs[0], cv.inputs["Y"])
        return cv.outputs["Vector"]

    uvn, uvf = uv(near), uv(far)

    def tex(name, vec, *, non_color=True, interp="Linear"):
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = _load(name, non_color=non_color)
        n.interpolation = interp
        n.extension = "EXTEND"
        nt.links.new(vec, n.inputs["Vector"])
        return n

    # near weight: 1 inside the near region, ramping to 0 over `feather` at its edges
    edges = []
    for val, sign in ((near[0], 1), (near[1], -1)):
        e = C.nmath(nt, "SUBTRACT", X, value_b=val) if sign > 0 else C.nmath(nt, "SUBTRACT", value_a=val, b=X)
        edges.append(e.outputs[0])
    for val, sign in ((near[2], 1), (near[3], -1)):
        e = C.nmath(nt, "SUBTRACT", Y, value_b=val) if sign > 0 else C.nmath(nt, "SUBTRACT", value_a=val, b=Y)
        edges.append(e.outputs[0])
    m = C.nmath(nt, "MINIMUM", edges[0], edges[1])
    m = C.nmath(nt, "MINIMUM", m.outputs[0], edges[2])
    m = C.nmath(nt, "MINIMUM", m.outputs[0], edges[3])
    wn = C.nmath(nt, "MULTIPLY", m.outputs[0], value_b=1.0 / feather, clamp=True).outputs[0]

    # --- albedo + relief -----------------------------------------------------------
    an = tex(p["albedo"]["near"]["file"], uvn, non_color=False)
    af = tex(p["albedo"]["far"]["file"], uvf, non_color=False)
    _, alb = C.mix_color(nt, wn, af.outputs["Color"], an.outputs["Color"])
    hs = nt.nodes.new("ShaderNodeHueSaturation")
    hs.inputs["Saturation"].default_value = saturation
    hs.inputs["Value"].default_value = albedo_gain
    nt.links.new(alb, hs.inputs["Color"])
    xn = tex(p["dem"]["near"]["aux"], uvn)
    xf = tex(p["dem"]["far"]["aux"], uvf)
    _, aux = C.mix_color(nt, wn, xf.outputs["Color"], xn.outputs["Color"])
    sc = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(aux, sc.inputs["Color"])
    # relief factor 1 + k (2 r - 1)
    rf = C.nmath(nt, "MULTIPLY_ADD", sc.outputs["Red"], value_b=2.0 * relief)
    rf.inputs[2].default_value = 1.0 - relief
    _, shaded = C.mix_color(nt, 1.0, hs.outputs["Color"], rf.outputs[0], "MULTIPLY")
    water = sc.outputs["Green"]
    _, base = C.mix_color(nt, water, shaded, (0.006, 0.012, 0.020, 1.0))
    nt.links.new(base, bsdf.inputs["Base Color"])
    _, rough = C.mix_float(nt, water, 0.95, 0.07)
    nt.links.new(rough, bsdf.inputs["Roughness"])
    _, spec = C.mix_float(nt, water, 0.12, 0.5)
    nt.links.new(spec, bsdf.inputs["Specular IOR Level"])

    # --- lamps -------------------------------------------------------------------
    ln = tex(L["near"]["file"], uvn)
    lf = tex(L["far"]["file"], uvf)

    def decode(node, dmax):
        s = nt.nodes.new("ShaderNodeSeparateColor")
        nt.links.new(node.outputs["Color"], s.inputs["Color"])
        out = []
        for ch, k in (("Red", dmax), ("Green", dmax), ("Blue", 1.0)):
            pw = C.nmath(nt, "POWER", s.outputs[ch], value_b=2.2)
            out.append(C.nmath(nt, "MULTIPLY", pw.outputs[0], value_b=k).outputs[0])
        return out

    dn_ = decode(ln, L["near"]["dmax"])
    df_ = decode(lf, L["far"]["dmax"])
    chans = [C.mix_float(nt, wn, f, n)[1] for f, n in zip(df_, dn_)]
    sodium, led = C.kelvin_rgb(2150), C.kelvin_rgb(3000)
    glow_c = C.kelvin_rgb(2300)

    def scale(rgb, fac, k):
        v = nt.nodes.new("ShaderNodeVectorMath")
        v.operation = "SCALE"
        v.inputs[0].default_value = tuple(c * k for c in rgb)
        nt.links.new(fac, v.inputs["Scale"])
        return v.outputs["Vector"]

    def add(a, b):
        v = nt.nodes.new("ShaderNodeVectorMath")
        v.operation = "ADD"
        nt.links.new(a, v.inputs[0])
        nt.links.new(b, v.inputs[1])
        return v.outputs["Vector"]

    em = add(add(scale(sodium, chans[0], lamp_strength), scale(led, chans[1], lamp_strength)),
             scale(glow_c, chans[2], glow_strength))
    nt.links.new(em, bsdf.inputs["Emission Color"])
    bsdf.inputs["Emission Strength"].default_value = 1.0
    return mat


# --------------------------------------------------------------------------- #
# overlays
# --------------------------------------------------------------------------- #
def _densify(pts, step, closed):
    out = []
    n = len(pts)
    m = n if closed else n - 1
    for i in range(m):
        (x0, y0), (x1, y1) = pts[i][:2], pts[(i + 1) % n][:2]
        k = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
        for t in range(k):
            out.append((x0 + (x1 - x0) * t / k, y0 + (y1 - y0) * t / k))
    if not closed:
        out.append(tuple(pts[-1][:2]))
    return out


def drape(pts, *, step=120.0, lift=6.0, closed=True):
    d = np.asarray(_densify(pts, step, closed))
    z = height(d[:, 0], d[:, 1]) + lift
    return [(float(a), float(b), float(c)) for (a, b), c in zip(d, z)]


def _line(col, name, pts, width, color, camera_strength, *, closed=True, strength=0.0):
    mat = C.emissive_material(name + "_m", color, strength, camera_strength=camera_strength)
    obj = C.ribbon_mesh(name, drape(pts, closed=closed), width, None, col=col, material=mat, closed=closed)
    for attr in ("visible_shadow", "visible_diffuse", "visible_glossy"):
        setattr(obj, attr, False)
    return obj


def _fill(col, name, poly, color, camera_strength):
    """A faint emissive fill: a fan from the centroid, draped."""
    mat = C.emissive_material(name + "_m", color, 0.0, camera_strength=camera_strength)
    ring = drape(poly, step=60.0, lift=4.0)
    cx = sum(p[0] for p in ring) / len(ring)
    cy = sum(p[1] for p in ring) / len(ring)
    cz = float(height(np.array([cx]), np.array([cy]))[0]) + 4.0
    verts = [(cx, cy, cz)] + ring
    faces = [(0, i + 1, (i + 1) % len(ring) + 1) for i in range(len(ring))]
    obj = C.mesh_object(name, verts, faces, col=col, material=mat)
    for attr in ("visible_shadow", "visible_diffuse", "visible_glossy"):
        setattr(obj, attr, False)
    return obj


def build_overlays(col, *, scale=1.0, uchicago=True):
    """Outlines at a width that survives 24 m pixels: about 1.5-2 px at 4K."""
    p = plan()
    w = 42.0 * scale
    # Fermilab: boundary, the proposed chain, the existing rings
    _line(col, "fnal_boundary", site.CAMPUS_BOUNDARY, w, BOUNDARY_WHITE, 4.0)
    _line(col, "mc_ring", C.circle_points(site.MC_C, site.MC_R, n=240), w * 1.1, site.COLLIDER_COLOR, 9.0)
    for name, c, r in site.RCS_RINGS:
        _line(col, f"{name}_ring", C.circle_points(c, r, n=240), w * 0.8, site.COLLIDER_COLOR, 4.5)
    _line(col, "tevatron_ring", C.circle_points(site.TEV_C, site.TEV_R, n=180), w * 0.8, EXISTING_AMBER, 4.0)
    _line(col, "mi_ring", C.circle_points(site.MI_C, site.MI_RX, n=120), w * 0.8, EXISTING_AMBER, 4.0)
    # Chicago: the city limits in the boundary vocabulary, a little lighter than
    # the site's -- it is context, the site is the subject
    for k, ring in enumerate(p["chicago_limits"]):
        _line(col, f"chicago_limits_{k}", ring, w * 0.7, BOUNDARY_WHITE, 1.5)
    if uchicago:
        for k, poly in enumerate(p["uchicago_campus"]):
            _line(col, f"uchicago_{k}", poly, w * 0.75, BOUNDARY_WHITE, 4.0)
            _fill(col, f"uchicago_fill_{k}", poly, BOUNDARY_WHITE, 0.55)
    print(f"[metro] overlays: Fermilab boundary + rings, Chicago limits "
          f"({p['chicago_area_km2']} km2), UChicago ({p['uchicago_area_km2']} km2)")


# --------------------------------------------------------------------------- #
# comparison rings
# --------------------------------------------------------------------------- #
# Each layout is a closed sequence of (length m, turn deg) segments, from the
# machines' design documents; a segment with turn 0 is a straight.
#   LHC: 8 arcs x 2803.9 m + 8 straights x 528 m = 26 659 m (JINST 3 S08001)
#   FCC: 8 arcs x 9616.175 m, straights alternating 1400 m (experimental) and
#        2032 m (technical) = 90 657.4 m (FCC Feasibility Study 2025, PA31)
#   SSC: 2 arcs x 35 280 m bending 168 deg (mean r 12 032 m) + 2 clusters x
#        8280 m bending a net 12 deg = 87 120 m (SSCL-SR-1041, lattice rev 4);
#        major axis N18.8W at the Texas site
#   VLHC Stage 1 (the transmission-line 'Pipetron' magnet): 2 half-circle arcs
#        of r 35 km bending 174.375 deg + 2 clusters of 9.75 km bending 5.625 deg
#        = 233 km (Fermilab-TM-2149, 2001); symmetry axis N-S through Fermilab
def _layout(name):
    if name == "lhc":
        return [(2803.9, 45.0), (528.0, 0.0)] * 8, 0.0, "LHC, 26.7 km"
    if name == "fcc":
        seq = []
        for k in range(4):
            seq += [(9616.175, 45.0), (1400.0, 0.0), (9616.175, 45.0), (2032.0, 0.0)]
        return seq, 0.0, "FCC, 90.7 km"
    if name == "ssc":
        return [(8280.0, 12.0), (35280.0, 168.0)] * 2, -18.8, "SSC, 87.1 km"
    if name == "vlhc":
        return [(9750.0, 5.625), (35000.0 * math.radians(174.375), 174.375)] * 2, 90.0, "VLHC Stage 1, 233 km"
    raise ValueError(name)


def ring_path(name, step=150.0):
    """Trace a layout heading north from the origin, turning left (counter-
    clockwise in plan); returns points centred on the bounding-box centre with
    the first segment's direction along +y, rotated by the layout's azimuth."""
    seq, az, label = _layout(name)
    x = y = 0.0
    h = math.pi / 2                       # heading: +y
    pts = [(x, y)]
    for length, turn in seq:
        n = max(1, int(length / step))
        dl, dh = length / n, math.radians(turn) / n
        for _ in range(n):
            if dh:
                h += dh / 2
                x += dl * math.cos(h)
                y += dl * math.sin(h)
                h += dh / 2
            else:
                x += dl * math.cos(h)
                y += dl * math.sin(h)
            pts.append((x, y))
    gap = math.hypot(pts[-1][0], pts[-1][1])
    pts = pts[:-1]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    a = math.radians(-az)                 # azimuth clockwise from north -> plan rotation
    ca, sa = math.cos(a), math.sin(a)
    out = [((px - cx) * ca - (py - cy) * sa, (px - cx) * sa + (py - cy) * ca) for px, py in pts]
    total = sum(s[0] for s in seq)
    return out, label, total, gap


def middle():
    """Halfway between the muon collider's centre and Willis Tower."""
    wx, wy = plan()["willis_tower"]
    return ((site.MC_C[0] + wx) / 2.0, (site.MC_C[1] + wy) / 2.0)


def build_comparison(col, name, *, at="middle", scale=1.0):
    pts, label, total, gap = ring_path(name)
    if at == "proposed" and name == "vlhc":
        # TM-2149's preferred North Ring: symmetry axis N-S through Fermilab, the
        # Fermilab cluster at its south end, running E-W and tangent to the
        # Tevatron (taken here as tangent on the Tevatron's north side).
        ymin = min(p[1] for p in pts)
        cx = site.TEV_C[0]
        cy = site.TEV_C[1] + site.TEV_R - ymin
    else:
        cx, cy = middle()
    pts = [(x + cx, y + cy) for x, y in pts]
    _line(col, f"compare_{name}", pts, 46.0 * scale, COMPARE_VIOLET, 1.6)
    print(f"[metro] comparison ring {label}: traced {total / 1000:.2f} km "
          f"(closure gap {gap:.0f} m), centred ({cx / 1000:.1f}, {cy / 1000:.1f}) km [{at}]")
    return label


# --------------------------------------------------------------------------- #
# atmosphere
# --------------------------------------------------------------------------- #
def build_haze(col, *, density=1.4e-4, scale_height=520.0, center=(30000.0, 70000.0),
               size=(420000.0, 320000.0), zlo=-4000.0, zhi=3500.0, patch_scale=7000.0,
               patch=0.35, anisotropy=0.35, seed=7.0):
    """atmosphere.build_haze's aerial haze -- same density, falloff, colour and
    patchiness -- but measured from the curved ground rather than z = 0, and in
    a box that covers the whole metro footprint. A flat slab would leave the far
    ground, which sits 2 km below the tangent plane, outside the air."""
    mat = bpy.data.materials.new("metro_haze")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.remove(nt.nodes["Principled BSDF"])
    out = nt.nodes["Material Output"]
    geo_n = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geo_n.outputs["Position"], sep.inputs["Vector"])
    x2 = C.nmath(nt, "MULTIPLY", sep.outputs["X"], sep.outputs["X"])
    y2 = C.nmath(nt, "MULTIPLY", sep.outputs["Y"], sep.outputs["Y"])
    r2 = C.nmath(nt, "ADD", x2.outputs[0], y2.outputs[0])
    alt = C.nmath(nt, "MULTIPLY_ADD", r2.outputs[0], value_b=1.0 / (2.0 * G.R_EARTH))
    nt.links.new(sep.outputs["Z"], alt.inputs[2])
    zc = C.nmath(nt, "MAXIMUM", alt.outputs[0], value_b=0.0)
    zn = C.nmath(nt, "DIVIDE", zc.outputs[0], value_b=-scale_height)
    ex = C.nmath(nt, "EXPONENT", zn.outputs[0])
    n1 = nt.nodes.new("ShaderNodeTexNoise")
    n1.inputs["Scale"].default_value = 1.0 / patch_scale
    n1.inputs["Detail"].default_value = 2.0
    n1.inputs["Roughness"].default_value = 0.5
    n1.noise_dimensions = "4D"
    n1.inputs["W"].default_value = seed
    nt.links.new(geo_n.outputs["Position"], n1.inputs["Vector"])
    band = C.nmath(nt, "MULTIPLY_ADD", n1.outputs["Fac"], value_b=2.0 * patch)
    band.inputs[2].default_value = 1.0 - patch
    d = C.nmath(nt, "MULTIPLY", ex.outputs[0], band.outputs[0])
    d = C.nmath(nt, "MULTIPLY", d.outputs[0], value_b=density)
    vol = nt.nodes.new("ShaderNodeVolumePrincipled")
    nt.links.new(d.outputs[0], vol.inputs["Density"])
    vol.inputs["Anisotropy"].default_value = anisotropy
    vol.inputs["Color"].default_value = (0.72, 0.82, 1.0, 1.0)
    nt.links.new(vol.outputs["Volume"], out.inputs["Volume"])
    mat.cycles.volume_sampling = "EQUIANGULAR"
    bm = C.box_bmesh(size[0], size[1], zhi - zlo)
    obj = C.bmesh_object("metro_haze", bm, col=col, material=mat, location=(center[0], center[1], zlo))
    obj.visible_shadow = False
    return obj
