#!/usr/bin/env python3
"""Fetch the real Fermilab site layout from OpenStreetMap into a local frame.

The scene used to be built from hand-guessed positions. This pulls the actual
geometry -- site boundary, roads, water, woods, buildings and the accelerator
features that are mapped -- and projects it to metres in the frame `scene/`
uses (x east, y north, origin at Wilson Hall), writing
`assets/geo/fermilab_site.json`.

OSM data is ODbL: attribute "(c) OpenStreetMap contributors" wherever the
render is published.

Usage:
    python3 tools/fetch_geodata.py            # fetch everything, write the JSON
    python3 tools/fetch_geodata.py --dry-run  # show what each query returns
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(ROOT, "assets", "geo")
OUT = os.path.join(OUT_DIR, "fermilab_site.json")

MIRRORS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
]

# Wilson Hall, the origin of the scene's metric frame.
ORIGIN_LAT, ORIGIN_LON = 41.83203, -88.26136
# Generous box: the site plus enough surroundings for the distance view.
BBOX = (41.76, -88.36, 41.92, -88.15)          # S, W, N, E

SITE_WAY = 31974155                             # Fermi National Accelerator Laboratory

# (key, Overpass query body, what we keep)
QUERIES = {
    "site": f"way({SITE_WAY}); out geom;",
    "accel": (
        'way["name"~"Tevatron|Main Injector|Booster|Linac|Recycler|Ring Road",i]'
        f'({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]}); out geom;'
    ),
    "water": (
        f'(way["natural"="water"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});'
        f'way["water"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});); out geom;'
    ),
    "wood": (
        f'(way["natural"="wood"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});'
        f'way["landuse"="forest"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});); out geom;'
    ),
    "major_roads": (
        'way["highway"~"^(motorway|trunk|primary|secondary|tertiary)$"]'
        f'({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]}); out geom;'
    ),
    "site_roads": (
        'way["highway"~"^(residential|unclassified|service)$"]'
        "(41.81,-88.29,41.88,-88.20); out geom;"
    ),
    "buildings": 'way["building"](41.815,-88.285,41.875,-88.205); out geom;',
    "urban": (
        f'way["landuse"~"^(residential|commercial|retail|industrial)$"]'
        f'({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]}); out geom;'
    ),
}


def overpass(body: str, *, timeout=180, tries=3) -> dict:
    q = f"[out:json][timeout:{timeout}];\n{body}\n"
    last = None
    for attempt in range(tries):
        for url in MIRRORS:
            try:
                req = urllib.request.Request(
                    url, data=q.encode(), headers={"User-Agent": "FermilabMuCBlender/1.0 (scene geodata)"})
                with urllib.request.urlopen(req, timeout=timeout + 30) as r:  # noqa: S310
                    return json.loads(r.read().decode())
            except Exception as e:  # noqa: BLE001
                last = f"{url.split('/')[2]}: {e}"
                print(f"    {last}")
                time.sleep(4)
        wait = 15 * (attempt + 1)
        print(f"    all mirrors failed; waiting {wait}s")
        time.sleep(wait)
    raise RuntimeError(f"Overpass failed: {last}")


def to_local(lat: float, lon: float) -> tuple[float, float]:
    """Equirectangular projection about Wilson Hall: x east, y north, metres.

    Over a 20 km box the distortion against a proper projection is centimetres,
    far below anything visible here.
    """
    k = 111320.0
    x = (lon - ORIGIN_LON) * k * math.cos(math.radians(ORIGIN_LAT))
    y = (lat - ORIGIN_LAT) * k
    return x, y


def ways(res: dict) -> list[dict]:
    out = []
    for e in res.get("elements", []):
        g = e.get("geometry") or []
        if len(g) < 2:
            continue
        pts = [to_local(p["lat"], p["lon"]) for p in g]
        t = e.get("tags", {})
        out.append({
            "id": e["id"],
            "name": t.get("name", ""),
            "tags": {k: v for k, v in t.items()
                     if k in ("highway", "building", "natural", "water", "landuse", "name")},
            "closed": abs(pts[0][0] - pts[-1][0]) < 1 and abs(pts[0][1] - pts[-1][1]) < 1,
            "pts": [[round(x, 1), round(y, 1)] for x, y in pts],
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    data: dict[str, list] = {}
    for key, body in QUERIES.items():
        print(f"[geo] {key} ...")
        res = overpass(body)
        w = ways(res)
        data[key] = w
        xs = [p[0] for e in w for p in e["pts"]] or [0]
        ys = [p[1] for e in w for p in e["pts"]] or [0]
        print(f"[geo]   {len(w)} ways, x {min(xs):.0f}..{max(xs):.0f} m, y {min(ys):.0f}..{max(ys):.0f} m")
        time.sleep(3)          # be a good Overpass citizen

    if a.dry_run:
        for k, v in data.items():
            named = [e["name"] for e in v if e["name"]][:8]
            print(f"{k:14s} {len(v):4d} ways  e.g. {named}")
        return 0

    os.makedirs(OUT_DIR, exist_ok=True)
    payload = {
        "origin": {"lat": ORIGIN_LAT, "lon": ORIGIN_LON, "note": "Wilson Hall; x east, y north, metres"},
        "bbox": BBOX,
        "attribution": "(c) OpenStreetMap contributors, ODbL",
        "layers": data,
    }
    with open(OUT, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    print(f"[geo] wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
