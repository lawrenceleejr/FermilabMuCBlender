#!/usr/bin/env python3
"""One-time geodata snapshot for the Fermilab scene.

Downloads AWS terrarium terrain tiles and OSM building footprints, converts
everything to the local ENU frame (meters, origin at Wilson Hall), and writes
the committed snapshot under data/:

    data/heightmap.npz         regular ENU elevation grid + georef metadata
    data/buildings.geojson     footprints with height/name/highlight props
    data/site_boundary.geojson Fermilab site boundary polygon

Run from the repo root:  python3 scripts/fetch_data.py
CI never runs this — it consumes the committed outputs only.
"""

import argparse
import io
import json
import math
import os
import sys
import time

import numpy as np
import requests
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import facility
from scripts import geo

TERRAIN_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
HEADERS = {"User-Agent": "FermilabMuCBlender/0.1 (one-time data snapshot)"}
TILE_SIZE = 256


def http_get(url, retries=3, **kw):
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=60, **kw)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {retries} tries: {last}")


# --- Terrain -----------------------------------------------------------------

def fetch_terrain(bbox, zoom, grid_n):
    south, west, north, east = bbox
    x0f, y1f = geo.latlon_to_tile(south, west, zoom)   # note: y grows southward
    x1f, y0f = geo.latlon_to_tile(north, east, zoom)
    tx0, tx1 = int(x0f), int(x1f)
    ty0, ty1 = int(y0f), int(y1f)
    nx, ny = tx1 - tx0 + 1, ty1 - ty0 + 1
    print(f"terrain: {nx}x{ny} tiles at z={zoom}")

    mosaic = np.zeros((ny * TILE_SIZE, nx * TILE_SIZE), dtype=np.float64)
    for ix in range(tx0, tx1 + 1):
        for iy in range(ty0, ty1 + 1):
            r = http_get(TERRAIN_URL.format(z=zoom, x=ix, y=iy))
            img = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"),
                             dtype=np.float64)
            elev = img[:, :, 0] * 256.0 + img[:, :, 1] + img[:, :, 2] / 256.0 - 32768.0
            mosaic[(iy - ty0) * TILE_SIZE:(iy - ty0 + 1) * TILE_SIZE,
                   (ix - tx0) * TILE_SIZE:(ix - tx0 + 1) * TILE_SIZE] = elev

    # ENU bounds of the requested bbox
    xw, ys = geo.latlon_to_enu(south, west, facility.ORIGIN_LAT, facility.ORIGIN_LON)
    xe, yn = geo.latlon_to_enu(north, east, facility.ORIGIN_LAT, facility.ORIGIN_LON)

    # Resample the Mercator mosaic onto a regular ENU grid (bilinear).
    xs_grid = np.linspace(xw, xe, grid_n)
    ys_grid = np.linspace(ys, yn, grid_n)
    Xg, Yg = np.meshgrid(xs_grid, ys_grid)  # (grid_n, grid_n), row 0 = south
    lat = facility.ORIGIN_LAT + Yg / geo.M_PER_DEG_LAT
    lon = facility.ORIGIN_LON + Xg / (geo.M_PER_DEG_LON_EQ *
                                      math.cos(math.radians(facility.ORIGIN_LAT)))

    # lat/lon -> fractional pixel coordinates in the mosaic
    n = 2.0 ** zoom
    xt = (lon + 180.0) / 360.0 * n
    lat_r = np.radians(lat)
    yt = (1.0 - np.arcsinh(np.tan(lat_r)) / np.pi) / 2.0 * n
    px = (xt - tx0) * TILE_SIZE - 0.5
    py = (yt - ty0) * TILE_SIZE - 0.5

    px = np.clip(px, 0, mosaic.shape[1] - 1.001)
    py = np.clip(py, 0, mosaic.shape[0] - 1.001)
    ix0 = np.floor(px).astype(int)
    iy0 = np.floor(py).astype(int)
    fx = px - ix0
    fy = py - iy0
    z = (mosaic[iy0, ix0] * (1 - fx) * (1 - fy)
         + mosaic[iy0, ix0 + 1] * fx * (1 - fy)
         + mosaic[iy0 + 1, ix0] * (1 - fx) * fy
         + mosaic[iy0 + 1, ix0 + 1] * fx * fy)

    z_datum = float(np.mean(z))
    print(f"terrain: elevation range {z.min():.1f}..{z.max():.1f} m ASL, "
          f"datum (mean) {z_datum:.1f} m")
    return {
        "z": z.astype(np.float32),
        "x0": xw, "y0": ys,
        "dx": (xe - xw) / (grid_n - 1),
        "dy": (yn - ys) / (grid_n - 1),
        "z_datum": z_datum,
        "origin_latlon": np.array([facility.ORIGIN_LAT, facility.ORIGIN_LON]),
    }


# --- Buildings ---------------------------------------------------------------

def overpass_query(query):
    last = None
    for url in OVERPASS_ENDPOINTS:
        for attempt in range(3):
            try:
                r = requests.post(url, data={"data": query}, timeout=180,
                                  headers=HEADERS)
                r.raise_for_status()
                return r.json()
            except Exception as e:  # noqa: BLE001
                last = e
                print(f"overpass: {url} attempt {attempt + 1} failed: {e}")
                time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"all overpass endpoints failed: {last}")


def parse_height(tags):
    h = tags.get("height")
    if h:
        try:
            return float(str(h).lower().replace("m", "").strip())
        except ValueError:
            pass
    lv = tags.get("building:levels")
    if lv:
        try:
            return max(3.0, float(lv) * facility.METERS_PER_LEVEL)
        except ValueError:
            pass
    return facility.DEFAULT_BUILDING_HEIGHT


def ring_to_enu(coords):
    return [list(geo.latlon_to_enu(c["lat"], c["lon"],
                                   facility.ORIGIN_LAT, facility.ORIGIN_LON))
            for c in coords]


def fetch_buildings(bbox):
    s, w, n, e = bbox
    q = f"""
[out:json][timeout:180];
( way["building"]({s},{w},{n},{e});
  relation["building"]({s},{w},{n},{e}); );
out body geom;
"""
    data = overpass_query(q)
    feats = []
    wilson = None
    skipped = 0
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        rings = []
        if el["type"] == "way" and "geometry" in el:
            rings = [ring_to_enu(el["geometry"])]
        elif el["type"] == "relation":
            for m in el.get("members", []):
                if m.get("role") == "outer" and "geometry" in m:
                    rings.append(ring_to_enu(m["geometry"]))
        for ring in rings:
            # drop closing duplicate point
            if len(ring) > 1 and ring[0] == ring[-1]:
                ring = ring[:-1]
            if len(ring) < 3:
                skipped += 1
                continue
            name = tags.get("name", "")
            height = parse_height(tags)
            highlight = False
            cx = sum(p[0] for p in ring) / len(ring)
            cy = sum(p[1] for p in ring) / len(ring)
            if "wilson hall" in name.lower() and math.hypot(cx, cy) < 500.0:
                height = facility.WILSON_HALL_HEIGHT
                highlight = True
            feats.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {"height": height, "name": name,
                               "highlight": highlight},
            })
            if highlight:
                wilson = feats[-1]

    if wilson is None:
        print("buildings: Wilson Hall NOT found in OSM -> using fallback footprint")
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Polygon",
                         "coordinates": [[list(p) for p in
                                          facility.WILSON_HALL_FALLBACK]]},
            "properties": {"height": facility.WILSON_HALL_HEIGHT,
                           "name": "Wilson Hall (fallback)",
                           "highlight": True},
        })
    else:
        print(f"buildings: Wilson Hall found: {wilson['properties']['name']!r}")
    print(f"buildings: {len(feats)} footprints ({skipped} degenerate skipped)")
    return {"type": "FeatureCollection", "features": feats}


def fetch_boundary():
    q = """
[out:json][timeout:180];
( way["name"~"Fermi National Accelerator Laboratory"](41.7,-88.4,41.95,-88.1);
  relation["name"~"Fermi National Accelerator Laboratory"](41.7,-88.4,41.95,-88.1); );
out body geom;
"""
    try:
        data = overpass_query(q)
    except RuntimeError as e:
        print(f"boundary: overpass failed ({e}) -> fallback square")
        data = {"elements": []}
    best, best_area = None, 0.0
    for el in data.get("elements", []):
        rings = []
        if el["type"] == "way" and "geometry" in el:
            rings = [el["geometry"]]
        elif el["type"] == "relation":
            rings = [m["geometry"] for m in el.get("members", [])
                     if m.get("role") == "outer" and "geometry" in m]
        for raw in rings:
            ring = ring_to_enu(raw)
            a = abs(geo.polygon_area(ring))
            if a > best_area:
                best, best_area = ring, a
    if best is None:
        best = [list(p) for p in facility.SITE_BOUNDARY_FALLBACK]
        best_area = abs(geo.polygon_area(best))
        print("boundary: using fallback square")
    print(f"boundary: area {best_area / 1e6:.1f} km^2 (site is ~27.5 km^2)")
    return {"type": "FeatureCollection", "features": [{
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [best]},
        "properties": {"name": "Fermilab site boundary"},
    }]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoom", type=int, default=13)
    ap.add_argument("--grid", type=int, default=512)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    bbox = facility.FETCH_BBOX

    hm = fetch_terrain(bbox, args.zoom, args.grid)
    np.savez_compressed(os.path.join(args.out, "heightmap.npz"), **hm)

    buildings = fetch_buildings(bbox)
    with open(os.path.join(args.out, "buildings.geojson"), "w") as f:
        json.dump(buildings, f)

    boundary = fetch_boundary()
    with open(os.path.join(args.out, "site_boundary.geojson"), "w") as f:
        json.dump(boundary, f)

    print("done.")


if __name__ == "__main__":
    main()
