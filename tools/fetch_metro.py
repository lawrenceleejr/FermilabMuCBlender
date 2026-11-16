#!/usr/bin/env python3
"""Download everything the metro (Fermilab-to-Chicago) view is built from.

    python3 tools/fetch_metro.py            # all layers, cached; re-runs are free
    python3 tools/fetch_metro.py --layer imagery_near

Then `python3 tools/bake_metro.py` reduces the cache to assets/metro/.

Layers, all open data, all cached per tile / per file under assets/metro/cache/:

  dem_near / dem_far        AWS Terrain Tiles (terrarium), zoom 12 / 9
  imagery_near / imagery_far USGS The National Map, USGSImageryOnly, zoom 13 / 10
                            (US Government work, public domain)
  nightlights               NASA Black Marble 2016 (VIIRS DNB) via GIBS, zoom 8
                            (NASA Earth Observatory, public domain)
  tiger_roads               US Census TIGER/Line 2024 all-roads, one file per county
                            for the counties under the NEAR region (public domain)
  tiger_prisec              TIGER/Line 2024 primary + secondary roads, statewide,
                            for IL, IN, WI and MI (public domain)
  chicago                   City of Chicago boundary, OSM relation 122604, via
                            Nominatim (ODbL)
  uchicago                  University of Chicago campus: OSM landuse=university
                            polygons through OpenFreeMap vector tiles (ODbL)

Why TIGER for the street network rather than OSM: the lights need *every* street
across ~10 000 km^2, which from Overpass is a multi-hundred-MB query that public
instances refuse, and the Geofabrik extracts are a 1 GB detour through a PBF
toolchain. TIGER is the same network (OSM's US roads were imported from it),
public domain, and arrives as one small shapefile per county.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from scene import metro_geo as G  # noqa: E402

CACHE = os.path.join(ROOT, "assets", "metro", "cache")
UA = "FermilabMuCBlender-scene-builder/1.0 (open-data fetch; low rate)"

TILESETS = {
    "dem_near": ("https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png", 12, "NEAR", "png"),
    "dem_far": ("https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png", 9, "FAR", "png"),
    "imagery_near": ("https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}", 13, "NEAR", "jpg"),
    "imagery_far": ("https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}", 10, "FAR", "jpg"),
    "nightlights": ("https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/VIIRS_Black_Marble/default/2016-01-01/GoogleMapsCompatible_Level8/{z}/{y}/{x}.png", 8, "FAR", "png"),
}

# Counties under the NEAR region (state FIPS + county FIPS)
NEAR_COUNTIES = {
    "17031": "Cook IL", "17043": "DuPage IL", "17089": "Kane IL", "17197": "Will IL",
    "17093": "Kendall IL", "17037": "DeKalb IL", "17111": "McHenry IL", "17097": "Lake IL",
    "17063": "Grundy IL", "18089": "Lake IN", "18127": "Porter IN",
}
PRISEC_STATES = {"17": "IL", "18": "IN", "55": "WI", "26": "MI"}
TIGER = "https://www2.census.gov/geo/tiger/TIGER2024"
OFM_TILEJSON = "https://tiles.openfreemap.org/planet"


def get(url, *, tries=4, timeout=60):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if getattr(e, "code", None) == 404:
                raise
            if k == tries - 1:
                raise
            time.sleep(1.5 * (k + 1))


def deg2tile(lat, lon, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def tile_range(region, z):
    la0, la1, lo0, lo1 = G.region_latlon_bbox(getattr(G, region))
    x0, y0 = deg2tile(la1, lo0, z)
    x1, y1 = deg2tile(la0, lo1, z)
    return int(x0), int(x1), int(y0), int(y1)


def fetch_tiles(name, workers=8):
    url, z, region, ext = TILESETS[name]
    x0, x1, y0, y1 = tile_range(region, z)
    jobs = [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
    d = os.path.join(CACHE, name, str(z))
    os.makedirs(d, exist_ok=True)
    todo = [(x, y) for x, y in jobs if not os.path.exists(os.path.join(d, f"{x}_{y}.{ext}"))]
    print(f"[fetch] {name}: zoom {z}, tiles x {x0}..{x1} y {y0}..{y1} = {len(jobs)} ({len(todo)} to get)")

    def one(xy):
        x, y = xy
        p = os.path.join(d, f"{x}_{y}.{ext}")
        try:
            b = get(url.format(z=z, x=x, y=y))
        except Exception as e:  # noqa: BLE001
            return f"{x},{y}: {e}"
        with open(p + ".part", "wb") as fh:
            fh.write(b)
        os.replace(p + ".part", p)
        return None

    fails = 0
    with cf.ThreadPoolExecutor(workers) as ex:
        for i, err in enumerate(ex.map(one, todo)):
            if err:
                fails += 1
                print(f"[fetch]   {name} {err}")
            if (i + 1) % 200 == 0:
                print(f"[fetch]   {name} {i + 1}/{len(todo)}")
    meta = dict(z=z, x0=x0, x1=x1, y0=y0, y1=y1, ext=ext, url=url)
    with open(os.path.join(CACHE, name, "tiles.json"), "w") as fh:
        json.dump(meta, fh)
    print(f"[fetch] {name}: done, {fails} failures")
    return fails


def fetch_file(url, path):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    b = get(url, timeout=300)
    with open(path + ".part", "wb") as fh:
        fh.write(b)
    os.replace(path + ".part", path)


def fetch_tiger_roads():
    for fips, nm in NEAR_COUNTIES.items():
        p = os.path.join(CACHE, "tiger", f"tl_2024_{fips}_roads.zip")
        print(f"[fetch] TIGER roads {nm} ({fips})")
        fetch_file(f"{TIGER}/ROADS/tl_2024_{fips}_roads.zip", p)


def fetch_tiger_prisec():
    for st, nm in PRISEC_STATES.items():
        p = os.path.join(CACHE, "tiger", f"tl_2024_{st}_prisecroads.zip")
        print(f"[fetch] TIGER primary/secondary roads {nm}")
        fetch_file(f"{TIGER}/PRISECROADS/tl_2024_{st}_prisecroads.zip", p)


def fetch_chicago():
    p = os.path.join(CACHE, "chicago_r122604.geojson")
    if os.path.exists(p):
        return
    q = urllib.parse.urlencode(dict(osm_ids="R122604", format="geojson", polygon_geojson=1))
    b = get(f"https://nominatim.openstreetmap.org/lookup?{q}")
    d = json.loads(b)
    assert d["features"], "Nominatim returned no feature for R122604"
    with open(p, "w") as fh:
        json.dump(d, fh)
    print(f"[fetch] Chicago boundary: {d['features'][0]['geometry']['type']}")


def fetch_uchicago():
    """landuse=university polygons from the vector tiles around the main quad."""
    p = os.path.join(CACHE, "uchicago_landuse.json")
    if os.path.exists(p):
        return
    import mapbox_vector_tile as mvt
    tj = json.loads(get(OFM_TILEJSON))
    url = tj["tiles"][0]
    z = 14
    lat, lon = G.UCHICAGO_QUAD
    fx, fy = deg2tile(lat, lon, z)
    polys = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            x, y = int(fx) + dx, int(fy) + dy
            b = get(url.format(z=z, x=x, y=y))
            try:
                b = gzip.decompress(b)
            except OSError:
                pass
            t = mvt.decode(b)
            for f in t.get("landuse", {}).get("features", []):
                if f["properties"].get("class") != "university":
                    continue
                g = f["geometry"]
                rings = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
                for poly in rings:
                    ll = []
                    for ring in poly[:1]:                  # outer ring only
                        for u, v in ring:
                            # tile coords (extent 4096, y up) -> lon/lat
                            tx, ty = x + u / 4096.0, y + 1.0 - v / 4096.0
                            n = 2 ** z
                            lo = tx / n * 360.0 - 180.0
                            la = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ty / n))))
                            ll.append((la, lo))
                    polys.append(ll)
    with open(p, "w") as fh:
        json.dump({"source": "OpenFreeMap vector tiles (OpenMapTiles schema), landuse class=university; "
                             "(c) OpenStreetMap contributors, ODbL", "zoom": z, "polygons": polys}, fh)
    print(f"[fetch] UChicago: {len(polys)} landuse=university polygon pieces")


LAYERS = {
    "dem_near": lambda: fetch_tiles("dem_near"),
    "dem_far": lambda: fetch_tiles("dem_far"),
    "nightlights": lambda: fetch_tiles("nightlights"),
    "imagery_far": lambda: fetch_tiles("imagery_far"),
    "imagery_near": lambda: fetch_tiles("imagery_near"),
    "tiger_prisec": fetch_tiger_prisec,
    "tiger_roads": fetch_tiger_roads,
    "chicago": fetch_chicago,
    "uchicago": fetch_uchicago,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", action="append", choices=sorted(LAYERS))
    a = ap.parse_args()
    bad = 0
    for name in (a.layer or list(LAYERS)):
        try:
            r = LAYERS[name]()
            bad += bool(r)
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"[fetch] {name} FAILED: {e}")
    print(f"[fetch] finished, {bad} layer(s) with problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
