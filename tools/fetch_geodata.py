#!/usr/bin/env python3
"""Fetch the real Fermilab site layout from OpenStreetMap into a local frame.

The scene used to be built from hand-guessed positions. This pulls the actual
geometry -- site boundary, roads, water, woods, buildings and the accelerator
features that are mapped -- and projects it to metres in the frame `scene/`
uses (x east, y north, origin at Wilson Hall), writing
`assets/geo/fermilab_site.json`.

OSM data is ODbL: attribute "(c) OpenStreetMap contributors" wherever the
render is published.

Every layer is cached separately in assets/geo/_layer_<name>.json, so a
timeout on one layer never discards the ones already fetched.

Usage:
    python3 tools/fetch_geodata.py --essential      # the layers the scene needs
    python3 tools/fetch_geodata.py --layer water    # just one, ignoring the rest
    python3 tools/fetch_geodata.py --merge          # rebuild the combined JSON
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

# Overpass times out on a wide box crossed with a broad tag, so every layer
# gets the smallest box that still covers what the camera sees, and each is a
# single tag rather than a union -- one query, one tag, one box. The wide box
# is only used for the layers that genuinely need the distance view.
SITE_BOX = (41.815, -88.295, 41.875, -88.205)   # the campus and its verges
NEAR_BOX = (41.78, -88.33, 41.90, -88.18)       # near surroundings
WIDE_BOX = BBOX                                 # the whole distance view


def _bb(box):
    return f"{box[0]},{box[1]},{box[2]},{box[3]}"


# (key, Overpass query body). One tag per query: a union of two broad tags over
# a 20 km box is what was producing 504s.
QUERIES = {
    "site": f"way({SITE_WAY}); out geom;",
    "accel": (
        'way["name"~"Tevatron|Main Injector|Booster|Linac|Recycler|Ring Road",i]'
        f'({_bb(NEAR_BOX)}); out geom;'
    ),
    "water": f'way["natural"="water"]({_bb(NEAR_BOX)}); out geom;',
    "wood": f'way["natural"="wood"]({_bb(NEAR_BOX)}); out geom;',
    "forest": f'way["landuse"="forest"]({_bb(NEAR_BOX)}); out geom;',
    "major_roads": (
        'way["highway"~"^(motorway|trunk|primary|secondary)$"]'
        f'({_bb(WIDE_BOX)}); out geom;'
    ),
    "minor_roads": (
        'way["highway"~"^(tertiary|residential|unclassified)$"]'
        f'({_bb(NEAR_BOX)}); out geom;'
    ),
    "site_roads": f'way["highway"="service"]({_bb(SITE_BOX)}); out geom;',
    "buildings": f'way["building"]({_bb(SITE_BOX)}); out geom;',
    "urban": (
        'way["landuse"~"^(residential|commercial|retail|industrial)$"]'
        f'({_bb(NEAR_BOX)}); out geom;'
    ),
}

# Layers where an empty result means the query failed rather than that the area
# is genuinely empty. One mirror answers 200 with zero elements when it is
# overloaded, which silently wrote an empty layer on the first run; treating
# empty as a failure here is what turns that into a retry.
NONEMPTY = {"site", "accel", "water", "wood", "major_roads", "minor_roads",
            "site_roads", "buildings"}

# Layers the scene actually needs to be rebuilt; the rest are nice to have.
ESSENTIAL = ["site", "accel", "water", "wood", "forest", "major_roads"]


def cache_path(key: str) -> str:
    return os.path.join(OUT_DIR, f"_layer_{key}.json")


def overpass(body: str, *, timeout=180, tries=3, nonempty=False) -> dict:
    q = f"[out:json][timeout:{timeout}];\n{body}\n"
    last = None
    for attempt in range(tries):
        for url in MIRRORS:
            try:
                req = urllib.request.Request(
                    url, data=q.encode(), headers={"User-Agent": "FermilabMuCBlender/1.0 (scene geodata)"})
                with urllib.request.urlopen(req, timeout=timeout + 30) as r:  # noqa: S310
                    res = json.loads(r.read().decode())
                if nonempty and not res.get("elements"):
                    raise RuntimeError("200 but no elements (mirror is shedding load)")
                return res
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


def summarise(key: str, w: list) -> None:
    xs = [p[0] for e in w for p in e["pts"]] or [0]
    ys = [p[1] for e in w for p in e["pts"]] or [0]
    print(f"[geo]   {len(w)} ways, x {min(xs):.0f}..{max(xs):.0f} m, "
          f"y {min(ys):.0f}..{max(ys):.0f} m")


def fetch_layer(key: str, *, refetch=False) -> list:
    """One layer, cached on disk.

    The first run lost 3 242 fetched roads because the process died on a later
    layer and nothing was written until the very end. Each layer is now its own
    file, so a failure costs only the layer that failed.
    """
    cp = cache_path(key)
    if os.path.exists(cp) and not refetch:
        w = json.load(open(cp))
        print(f"[geo] {key}: cached ({len(w)} ways)")
        return w
    print(f"[geo] {key} ...")
    res = overpass(QUERIES[key], nonempty=key in NONEMPTY)
    w = ways(res)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(cp, "w") as fh:
        json.dump(w, fh, separators=(",", ":"))
    summarise(key, w)
    return w


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--layer", action="append", default=None,
                    help="fetch only this layer (repeatable); default is all of them")
    ap.add_argument("--essential", action="store_true",
                    help=f"only the layers the scene needs: {', '.join(ESSENTIAL)}")
    ap.add_argument("--refetch", action="store_true", help="ignore the per-layer cache")
    ap.add_argument("--merge", action="store_true",
                    help="write the combined JSON from whatever layers are cached")
    a = ap.parse_args()

    if a.layer:
        keys = [k for k in a.layer if k in QUERIES]
        bad = [k for k in a.layer if k not in QUERIES]
        if bad:
            print(f"[geo] unknown layer(s) {bad}; known: {', '.join(QUERIES)}")
            return 2
    elif a.essential:
        keys = list(ESSENTIAL)
    else:
        keys = list(QUERIES)

    data: dict[str, list] = {}
    failed = []
    if not a.merge:
        for key in keys:
            try:
                data[key] = fetch_layer(key, refetch=a.refetch)
            except Exception as e:                        # noqa: BLE001
                print(f"[geo] {key}: FAILED -- {e}")
                failed.append(key)
                continue
            time.sleep(3)                                # be a good Overpass citizen

    if a.dry_run:
        for k, v in data.items():
            named = [e["name"] for e in v if e["name"]][:8]
            print(f"{k:14s} {len(v):4d} ways  e.g. {named}")
        return 0

    # merge every layer that is on disk, so a partial run still produces a
    # usable file and a later run only has to fill the gaps
    for key in QUERIES:
        if key in data:
            continue
        cp = cache_path(key)
        if os.path.exists(cp):
            data[key] = json.load(open(cp))

    os.makedirs(OUT_DIR, exist_ok=True)
    payload = {
        "origin": {"lat": ORIGIN_LAT, "lon": ORIGIN_LON, "note": "Wilson Hall; x east, y north, metres"},
        "bbox": BBOX,
        "attribution": "(c) OpenStreetMap contributors, ODbL",
        "layers": data,
    }
    with open(OUT, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    counts = ", ".join(f"{k} {len(v)}" for k, v in sorted(data.items()))
    print(f"[geo] wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB): {counts}")
    if failed:
        print(f"[geo] still missing: {', '.join(failed)} -- rerun with --layer for each")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
