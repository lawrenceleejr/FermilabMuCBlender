#!/usr/bin/env python3
"""Fetch a real elevation model for the Fermilab area as a Blender heightmap.

Source: AWS "Terrain Tiles" (terrarium PNG encoding), public domain / open
data, built from SRTM, NED and other national datasets. Elevation in metres is
  h = R * 256 + G + B / 256 - 32768

Northern Illinois is glacial plain: sampling a 50 km box around the site gives
179-284 m, i.e. about 105 m of total relief, rising gently towards the moraines
to the north-west. There are no mountains to put on the horizon -- what the
distance view actually needs is this low, long-wavelength relief, which is why
using the real DEM matters more here than it would in a mountain scene.

Writes assets/geo/dem.png (16-bit greyscale) and assets/geo/dem.json with the
metric extent and the elevation range the image maps to.

Usage:
    python3 tools/fetch_dem.py [--zoom 11] [--span 30000]
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import urllib.request

import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(ROOT, "assets", "geo")

ORIGIN_LAT, ORIGIN_LON = 41.83203, -88.26136       # Wilson Hall
TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"


def deg2tile(lat, lon, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    lr = math.radians(lat)
    y = (1.0 - math.log(math.tan(lr) + 1.0 / math.cos(lr)) / math.pi) / 2.0 * n
    return x, y


def tile2deg(x, y, z):
    n = 2 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


def fetch_tile(z, x, y):
    url = TILE_URL.format(z=z, x=x, y=y)
    with urllib.request.urlopen(url, timeout=120) as r:  # noqa: S310
        return Image.open(io.BytesIO(r.read())).convert("RGB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zoom", type=int, default=11)
    ap.add_argument("--span", type=float, default=30000.0, help="half-width of the output, metres")
    ap.add_argument("--size", type=int, default=2048, help="output heightmap pixels")
    a = ap.parse_args()

    z = a.zoom
    # tiles covering the requested span
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(ORIGIN_LAT))
    dlat = a.span / m_per_deg_lat
    dlon = a.span / m_per_deg_lon
    x0f, y0f = deg2tile(ORIGIN_LAT + dlat, ORIGIN_LON - dlon, z)
    x1f, y1f = deg2tile(ORIGIN_LAT - dlat, ORIGIN_LON + dlon, z)
    tx0, tx1 = int(math.floor(min(x0f, x1f))), int(math.floor(max(x0f, x1f)))
    ty0, ty1 = int(math.floor(min(y0f, y1f))), int(math.floor(max(y0f, y1f)))
    print(f"[dem] zoom {z}, tiles x {tx0}..{tx1}, y {ty0}..{ty1} "
          f"({(tx1 - tx0 + 1) * (ty1 - ty0 + 1)} tiles)")

    # stitch
    first = fetch_tile(z, tx0, ty0)
    tw, th = first.size
    mosaic = Image.new("RGB", (tw * (tx1 - tx0 + 1), th * (ty1 - ty0 + 1)))
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            im = first if (tx, ty) == (tx0, ty0) else fetch_tile(z, tx, ty)
            mosaic.paste(im, ((tx - tx0) * tw, (ty - ty0) * th))
            print(f"[dem]   tile {tx},{ty}")
    arr = np.asarray(mosaic).astype(np.float64)
    elev = arr[..., 0] * 256.0 + arr[..., 1] + arr[..., 2] / 256.0 - 32768.0

    # crop to the requested metric square around the origin
    lat_nw, lon_nw = tile2deg(tx0, ty0, z)
    lat_se, lon_se = tile2deg(tx1 + 1, ty1 + 1, z)
    H, W = elev.shape

    def px_of(lat, lon):
        u = (lon - lon_nw) / (lon_se - lon_nw) * W
        # latitude is Mercator-spaced; invert through the tile mapping
        yy = deg2tile(lat, lon, z)[1]
        v = (yy - ty0) / (ty1 + 1 - ty0) * H
        return u, v

    u0, v0 = px_of(ORIGIN_LAT + dlat, ORIGIN_LON - dlon)
    u1, v1 = px_of(ORIGIN_LAT - dlat, ORIGIN_LON + dlon)
    u0, u1 = sorted((u0, u1))
    v0, v1 = sorted((v0, v1))
    crop = elev[int(v0):int(v1) + 1, int(u0):int(u1) + 1]
    print(f"[dem] cropped to {crop.shape[1]}x{crop.shape[0]} px for a {2 * a.span / 1000:.0f} km square")

    lo, hi = float(np.nanmin(crop)), float(np.nanmax(crop))
    here = float(crop[crop.shape[0] // 2, crop.shape[1] // 2])
    print(f"[dem] elevation {lo:.0f}..{hi:.0f} m (relief {hi - lo:.0f} m); at the origin {here:.0f} m")
    print("[dem] for reference this is glacial plain -- no mountains, just moraine relief")

    # resample to the requested size and store as 16-bit
    img = Image.fromarray(crop.astype(np.float32), mode="F").resize((a.size, a.size), Image.BILINEAR)
    q = np.asarray(img)
    norm = (q - lo) / max(hi - lo, 1e-6)
    os.makedirs(OUT_DIR, exist_ok=True)
    Image.fromarray((norm * 65535).astype(np.uint16), mode="I;16").save(os.path.join(OUT_DIR, "dem.png"))
    meta = {
        "origin": {"lat": ORIGIN_LAT, "lon": ORIGIN_LON},
        "half_span_m": a.span,
        "elev_min_m": lo,
        "elev_max_m": hi,
        "elev_at_origin_m": here,
        "size_px": a.size,
        "source": "AWS Terrain Tiles (terrarium), open data; SRTM/NED derived",
        "note": "16-bit PNG; height = elev_min + (px/65535) * (elev_max - elev_min)",
    }
    with open(os.path.join(OUT_DIR, "dem.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[dem] wrote {OUT_DIR}/dem.png and dem.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
