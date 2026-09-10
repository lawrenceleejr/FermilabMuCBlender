#!/usr/bin/env python3
"""Fetch the freely licensed (CC0) textures and HDRIs the scene uses.

Sources:
  * ambientCG  (https://ambientcg.com)  -- CC0 1.0 PBR texture sets
  * Poly Haven (https://polyhaven.com)  -- CC0 1.0 HDRIs

  * NASA/GSFC SVS (https://svs.gsfc.nasa.gov/4851) -- Deep Star Maps 2020, public domain

Usage:  python3 tools/fetch_assets.py [--hdri-res 2k|4k|8k] [--starmap-res 4k|8k|16k] [--only textures|hdri|starmap]
Files land in assets/textures/<Set>/, assets/hdri/ and assets/starmap/. Existing files are kept.
The Fermilab night-sky HDRI itself is then built with:
    blender -b --python tools/make_sky_hdri.py -- --res 8k
"""
from __future__ import annotations

import argparse
import io
import pathlib
import sys
import time
import urllib.error
import urllib.request
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEX_DIR = ROOT / "assets" / "textures"
HDRI_DIR = ROOT / "assets" / "hdri"
STARMAP_DIR = ROOT / "assets" / "starmap"
SVS_URL = "https://svs.gsfc.nasa.gov/vis/a000000/a004800/a004851/starmap_2020_{res}.exr"

# ambientCG texture sets: (asset id, resolution/format). 2K for surfaces near the
# camera, 1K for far/small surfaces.
TEXTURES = [
    ("Grass004", "2K-JPG"),        # prairie grass base
    ("Ground037", "2K-JPG"),       # dry ground / mown areas, mixed into grass
    ("Concrete034", "2K-JPG"),     # Wilson Hall board-formed concrete
    ("Concrete042A", "1K-JPG"),    # service buildings, berm retaining walls
    ("Asphalt031", "1K-JPG"),      # roads
]

# Poly Haven HDRIs (pure skies -- no ground, so the horizon is ours to fill).
HDRIS = [
    "qwantani_dusk_2_puresky",     # deep blue-hour dusk, soft clouds  (primary)
    "kloppenheim_06_puresky",      # violet dusk, alternate
    "qwantani_night_puresky",      # clear starry night, alternate
]


# ambientCG rejects urllib's default "Python-urllib/3.x" with a flat 403, on
# every retry, so the fetcher looked like a network problem when it was a
# header problem. Verified: no User-Agent gives 403, this one gives 200 and
# 39.9 MB. NASA/GSFC and Poly Haven serve either way. The same lesson is
# already applied in get_fonts.py and fetch_geodata.py; this file predated both
# and never got it.
UA = "Mozilla/5.0 (X11; Linux x86_64) FermilabMuCBlender/1.0 (+CC0 asset fetch)"


def _get(url: str, retries: int = 4) -> bytes:
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=600) as r:  # noqa: S310
                return r.read()
        except urllib.error.HTTPError as e:               # noqa: PERF203
            last = e
            if e.code in (401, 403, 404, 410):
                # a refusal, not congestion: retrying cannot change the answer,
                # and four escalating sleeps only bury the real cause
                print(f"  {e.code} {e.reason} -- not retrying (check the "
                      "User-Agent or the URL)")
                break
            wait = 2 ** (i + 1)
            print(f"  retry {i + 1}/{retries} after {e} (sleep {wait}s)")
            time.sleep(wait)
        except Exception as e:  # noqa: BLE001
            last = e
            wait = 2 ** (i + 1)
            print(f"  retry {i + 1}/{retries} after error: {e} (sleep {wait}s)")
            time.sleep(wait)
    raise RuntimeError(f"failed to fetch {url}: {last}")


def fetch_textures() -> None:
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    for asset, res in TEXTURES:
        dest = TEX_DIR / f"{asset}_{res}"
        if dest.exists() and any(dest.glob("*_Color.jpg")):
            print(f"[textures] {asset} {res}: present")
            continue
        url = f"https://ambientcg.com/get?file={asset}_{res}.zip"
        print(f"[textures] {asset} {res}: downloading {url}")
        data = _get(url)
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            z.extractall(dest)
        print(f"[textures] {asset}: {len(data) / 1e6:.1f} MB -> {dest}")


def fetch_hdris(res: str) -> None:
    HDRI_DIR.mkdir(parents=True, exist_ok=True)
    for hid in HDRIS:
        dest = HDRI_DIR / f"{hid}_{res}.hdr"
        if dest.exists():
            print(f"[hdri] {hid} {res}: present")
            continue
        url = f"https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/{res}/{hid}_{res}.hdr"
        print(f"[hdri] {hid} {res}: downloading {url}")
        data = _get(url)
        dest.write_bytes(data)
        print(f"[hdri] {hid}: {len(data) / 1e6:.1f} MB -> {dest}")


def fetch_starmap(res: str) -> None:
    STARMAP_DIR.mkdir(parents=True, exist_ok=True)
    dest = STARMAP_DIR / f"starmap_2020_{res}.exr"
    if dest.exists():
        print(f"[starmap] {res}: present")
        return
    url = SVS_URL.format(res=res)
    print(f"[starmap] downloading {url} (4k=36 MB, 8k=130 MB, 16k=443 MB)")
    dest.write_bytes(_get(url))
    print(f"[starmap] -> {dest}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hdri-res", default="4k", choices=["1k", "2k", "4k", "8k"])
    p.add_argument("--starmap-res", default="8k", choices=["4k", "8k", "16k"])
    p.add_argument("--only", choices=["textures", "hdri", "starmap"])
    a = p.parse_args()
    if a.only in (None, "textures"):
        fetch_textures()
    if a.only in (None, "hdri"):
        fetch_hdris(a.hdri_res)
    if a.only in (None, "starmap"):
        fetch_starmap(a.starmap_res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
