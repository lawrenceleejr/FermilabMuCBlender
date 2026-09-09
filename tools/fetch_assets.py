#!/usr/bin/env python3
"""Fetch the freely licensed (CC0) textures and HDRIs the scene uses.

Sources:
  * ambientCG  (https://ambientcg.com)  -- CC0 1.0 PBR texture sets
  * Poly Haven (https://polyhaven.com)  -- CC0 1.0 HDRIs

Usage:  python3 tools/fetch_assets.py [--hdri-res 2k|4k|8k] [--only textures|hdri]
Files land in assets/textures/<Set>/ and assets/hdri/. Existing files are kept.
"""
from __future__ import annotations

import argparse
import io
import pathlib
import sys
import time
import urllib.request
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEX_DIR = ROOT / "assets" / "textures"
HDRI_DIR = ROOT / "assets" / "hdri"

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


def _get(url: str, retries: int = 4) -> bytes:
    last = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=600) as r:
                return r.read()
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hdri-res", default="4k", choices=["1k", "2k", "4k", "8k"])
    p.add_argument("--only", choices=["textures", "hdri"])
    a = p.parse_args()
    if a.only in (None, "textures"):
        fetch_textures()
    if a.only in (None, "hdri"):
        fetch_hdris(a.hdri_res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
