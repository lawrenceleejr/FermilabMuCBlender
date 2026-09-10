#!/usr/bin/env python3
"""Turn the raw Overpass dump into the compact site plan the scene consumes.

`fetch_geodata.py` writes everything Overpass returns -- 3.3 MB across ten
layers, most of it detail invisible at overview scale. This reduces that to the
few hundred kB the render actually needs, and, unlike the raw dump, the result
is committed: otherwise a render on another machine would silently differ
depending on whether that machine had run the fetcher.

Three jobs beyond simplifying:

  * Re-origin. The scene's frame was pinned to a guessed Wilson Hall at
    41.83203 N, -88.26136 E. OSM's own Wilson Hall building sits 701 m north of
    that point, so every position in the scene carried a 701 m offset. The
    origin here is the OSM building's centroid.

  * Derive the existing rings from the mapped features that trace them, rather
    than from memory. The Tevatron comes from the pooled Outer Ring Road
    fragments; the Main Injector from the arc of the Main Injector Pond, which
    rings its berm. Both are least-squares circle fits with the residual
    reported, so the fit can be judged rather than trusted.

  * Size the RCS site-filler ring: the largest circle that fits inside the real
    boundary with a setback, which is the honest answer to "how big a ring
    could this site hold" and is not any IMCC number.

Usage:
    python3 tools/bake_geodata.py [--out assets/plan/site_plan.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "assets", "geo", "fermilab_site.json")
OUT = os.path.join(ROOT, "assets", "plan", "site_plan.json")

# The fetch frame's assumed origin, and OSM's Wilson Hall relative to it.
FETCH_LAT, FETCH_LON = 41.83203, -88.26136
WILSON_IN_FETCH = (-25.0, 701.0)        # measured from way 25473315's centroid

# What to keep, how hard to simplify it, and how far out it is worth keeping.
KEEP = {
    "site":        dict(tol=4.0,  radius=9000.0),
    "water":       dict(tol=12.0, radius=9000.0),
    "wood":        dict(tol=18.0, radius=14000.0),
    "forest":      dict(tol=18.0, radius=14000.0),
    "major_roads": dict(tol=20.0, radius=16000.0),
    "minor_roads": dict(tol=20.0, radius=7000.0),
    "site_roads":  dict(tol=10.0, radius=5000.0),
    # Buildings were omitted here at first, which is why the Village came out
    # with nothing in it: the builder read the baked plan, and the baked plan
    # had no footprints. A small tolerance, because these are 10-40 m boxes and
    # simplifying them at road tolerance would collapse them to slivers.
    "buildings":   dict(tol=2.0,  radius=6500.0),
    "urban":       dict(tol=30.0, radius=14000.0),
}
SETBACK = 250.0                          # tunnel setback from the property line


def to_scene(pts):
    """Fetch-frame metres -> scene metres about OSM's Wilson Hall."""
    ox, oy = WILSON_IN_FETCH
    return [[p[0] - ox, p[1] - oy] for p in pts]


def rdp(pts, tol):
    """Ramer-Douglas-Peucker. Keeps the shape a viewer can resolve and drops
    the vertices that only cost file size."""
    if len(pts) < 3:
        return list(pts)
    a, b = np.array(pts[0], float), np.array(pts[-1], float)
    ab = b - a
    n = np.linalg.norm(ab)
    P = np.array(pts, float)
    if n < 1e-9:
        d = np.linalg.norm(P - a, axis=1)
    else:
        d = np.abs(np.cross(np.broadcast_to(ab, P.shape), P - a)) / n
    i = int(d.argmax())
    if d[i] <= tol:
        return [list(pts[0]), list(pts[-1])]
    return rdp(pts[:i + 1], tol)[:-1] + rdp(pts[i:], tol)


def fit_circle(pts):
    P = np.array([[p[0], p[1]] for p in pts], float)
    A = np.c_[2 * P[:, 0], 2 * P[:, 1], np.ones(len(P))]
    (cx, cy, c), *_ = np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)
    r = math.sqrt(max(c + cx * cx + cy * cy, 0.0))
    rms = float(np.sqrt(((np.hypot(P[:, 0] - cx, P[:, 1] - cy) - r) ** 2).mean()))
    return float(cx), float(cy), float(r), rms


def point_in(poly, x, y) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            xi = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < xi:
                inside = not inside
    return inside


def dist_to_edges(poly, x, y) -> float:
    best = float("inf")
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 < 1e-9 else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / L2))
        best = min(best, math.hypot(x - (ax + t * dx), y - (ay + t * dy)))
    return best


def inscribed_circle(poly, *, coarse=140, refines=6):
    """Largest circle inside the polygon: grid search, then local refinement.

    This is what sizes the site-filler synchrotron. Any ring bigger than this
    leaves the property, so it is a real constraint of the site rather than a
    design choice.
    """
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    best = (0.0, 0.0, -1.0)
    lo_x, hi_x, lo_y, hi_y = min(xs), max(xs), min(ys), max(ys)
    step = max((hi_x - lo_x) / coarse, (hi_y - lo_y) / coarse)
    for i in range(coarse + 1):
        for j in range(coarse + 1):
            x = lo_x + (hi_x - lo_x) * i / coarse
            y = lo_y + (hi_y - lo_y) * j / coarse
            if not point_in(poly, x, y):
                continue
            d = dist_to_edges(poly, x, y)
            if d > best[2]:
                best = (x, y, d)
    for _ in range(refines):                       # tighten around the winner
        cx, cy, cr = best
        step *= 0.5
        for dx in (-step, 0.0, step):
            for dy in (-step, 0.0, step):
                x, y = cx + dx, cy + dy
                if not point_in(poly, x, y):
                    continue
                d = dist_to_edges(poly, x, y)
                if d > best[2]:
                    best = (x, y, d)
    return best


def area_km2(poly) -> float:
    a = 0.0
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        a += x0 * y1 - x1 * y0
    return abs(a) / 2.0 / 1e6


def bake_dem(out_dir, *, size=512):
    """Downsample the fetched DEM into a committed heightmap.

    Relief here is glacial-plain scale -- 85 m within 6 km of the site, 267 m
    across the whole 60 km box -- so 512 px over 60 km (117 m per pixel) holds
    every landform the camera can resolve from 2.6 km up, at a ninth of the
    file size.
    """
    src_png = os.path.join(ROOT, "assets", "geo", "dem.png")
    src_json = os.path.join(ROOT, "assets", "geo", "dem.json")
    if not (os.path.exists(src_png) and os.path.exists(src_json)):
        print("[bake] no DEM to bake; run tools/fetch_dem.py for terrain relief")
        return None
    from PIL import Image
    meta = json.load(open(src_json))
    im = Image.open(src_png)
    im = im.resize((size, size), Image.BILINEAR)
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, "dem.png")
    im.save(dest)
    # the fetch frame's origin elevation still applies: the re-origin is 701 m,
    # far below one DEM pixel of horizontal shift mattering to the elevation datum
    out = dict(file="dem.png", size_px=size, half_span_m=meta["half_span_m"],
               elev_min_m=meta["elev_min_m"], elev_max_m=meta["elev_max_m"],
               elev_at_origin_m=meta["elev_at_origin_m"],
               source=meta["source"],
               note="height = elev_min + (px/65535) * (elev_max - elev_min); "
                    "subtract elev_at_origin for scene z")
    print(f"[bake] DEM {size}x{size} over {2 * meta['half_span_m'] / 1000:.0f} km "
          f"({os.path.getsize(dest) / 1e6:.2f} MB), relief "
          f"{meta['elev_max_m'] - meta['elev_min_m']:.0f} m")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    if not os.path.exists(a.raw):
        print(f"[bake] {a.raw} missing; run tools/fetch_geodata.py first")
        return 1
    raw = json.load(open(a.raw))
    layers = raw["layers"]

    ox, oy = WILSON_IN_FETCH
    origin_lat = FETCH_LAT + oy / 111320.0
    origin_lon = FETCH_LON + ox / (111320.0 * math.cos(math.radians(FETCH_LAT)))
    print(f"[bake] origin moved to OSM Wilson Hall: {origin_lat:.5f}, {origin_lon:.5f} "
          f"({oy:+.0f} m north, {ox:+.0f} m east of the old guess)")

    out_layers: dict[str, list] = {}
    for key, cfg in KEEP.items():
        src = layers.get(key) or []
        kept = []
        for w in src:
            pts = to_scene(w["pts"])
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            if math.hypot(cx, cy) > cfg["radius"]:
                continue
            simp = rdp(pts, cfg["tol"]) if len(pts) > 2 else pts
            if len(simp) < 2:
                continue
            kept.append({
                "name": w.get("name", ""),
                "closed": w.get("closed", False),
                "pts": [[round(x, 1), round(y, 1)] for x, y in simp],
            })
        v_in = sum(len(w["pts"]) for w in src)
        v_out = sum(len(w["pts"]) for w in kept)
        out_layers[key] = kept
        print(f"[bake] {key:12s} {len(src):5d} -> {len(kept):5d} ways, "
              f"{v_in:6d} -> {v_out:6d} vertices")

    if not out_layers.get("site"):
        print("[bake] no site boundary in the raw data; cannot continue")
        return 1
    boundary = [[p[0], p[1]] for p in out_layers["site"][0]["pts"]]
    if boundary[0] == boundary[-1]:
        boundary = boundary[:-1]
    print(f"[bake] boundary {len(boundary)} vertices, area {area_km2(boundary):.2f} km2 "
          f"({area_km2(boundary) * 247.105:.0f} acres)")

    derived: dict[str, dict] = {}
    pools = {
        "tevatron": [p for w in layers.get("accel", []) if w["name"] == "Outer Ring Road"
                     for p in to_scene(w["pts"])],
        "main_injector": [p for w in layers.get("accel", []) if w["name"] == "Main Injector Pond"
                          for p in to_scene(w["pts"])],
    }
    # circumferences we hold to, from the machines themselves
    known_c = {"tevatron": 6283.2, "main_injector": 3319.4}
    for name, pool in pools.items():
        if len(pool) < 12:
            print(f"[bake] {name}: only {len(pool)} points, skipping the fit")
            continue
        cx, cy, r, rms = fit_circle(pool)
        r_true = known_c[name] / (2.0 * math.pi)
        derived[name] = dict(center=[round(cx, 1), round(cy, 1)], radius=round(r_true, 1),
                             fitted_radius=round(r, 1), fit_rms_m=round(rms, 1),
                             traced_from=("Outer Ring Road" if name == "tevatron"
                                          else "Main Injector Pond"), points=len(pool))
        print(f"[bake] {name:14s} centre ({cx:7.1f},{cy:7.1f})  fitted r {r:6.1f} "
              f"(rms {rms:5.1f} m over {len(pool)} pts) -> using r {r_true:6.1f} from "
              f"{known_c[name]:.0f} m circumference")

    fx, fy, fr = inscribed_circle(boundary)
    rr = max(fr - SETBACK, 0.0)
    derived["site_filler"] = dict(center=[round(fx, 1), round(fy, 1)],
                                  radius=round(rr, 1),
                                  max_radius=round(fr, 1), setback_m=SETBACK,
                                  circumference_m=round(2 * math.pi * rr, 0))
    print(f"[bake] site filler: largest inscribed circle r {fr:.0f} m at ({fx:.0f},{fy:.0f}); "
          f"with a {SETBACK:.0f} m setback r {rr:.0f} m, circumference "
          f"{2 * math.pi * rr / 1000:.2f} km")

    # a committed, downsampled DEM: the 2048 px fetch is 5.6 MB and gitignored,
    # so without this a render on another machine would be flat unless that
    # machine had run fetch_dem.py
    dem_meta = bake_dem(os.path.dirname(a.out))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    payload = {
        "origin": {"lat": round(origin_lat, 6), "lon": round(origin_lon, 6),
                   "note": "OSM Wilson Hall centroid; x east, y north, metres"},
        "attribution": "(c) OpenStreetMap contributors, ODbL",
        "boundary": [[round(x, 1), round(y, 1)] for x, y in boundary],
        "boundary_area_km2": round(area_km2(boundary), 2),
        "derived": derived,
        "dem": dem_meta,
        "layers": out_layers,
    }
    with open(a.out, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    print(f"[bake] wrote {a.out} ({os.path.getsize(a.out) / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
