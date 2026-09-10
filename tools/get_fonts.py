#!/usr/bin/env python3
"""Download Google Fonts as TTF for self-hosting (reproducible, no runtime CDN).

Adapted from the typography skill's fetcher, which pulls woff2 for the web.
Here the consumer is matplotlib/FreeType (tools/annotate.py), which needs
TrueType, so we ask the CSS2 API with an old User-Agent -- it then serves .ttf
instead of .woff2. All Google Fonts are OFL/Apache/UFL, safe to bundle.

Usage:
    python3 tools/get_fonts.py                       # the families this repo uses
    python3 tools/get_fonts.py "Inter:wght@400;700"  # anything else
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import urllib.parse
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FONT_DIR = os.path.join(ROOT, "assets", "fonts")

# An old UA makes the CSS2 API serve TrueType rather than woff2.
UA_TTF = "Mozilla/4.0"

# IBM Plex is a superfamily (sans + mono pair by construction) designed for
# technical and engineering contexts, and it holds up at label sizes.
DEFAULT_FAMILIES = [
    "IBM Plex Sans:wght@300;400;500;600",
    "IBM Plex Mono:wght@400;500",
]

FACE_RE = re.compile(r"@font-face\s*\{([^}]*)\}", re.S)
URL_RE = re.compile(r"url\(([^)]+)\)")


def _prop(block: str, key: str) -> str | None:
    m = re.search(rf"{key}:\s*([^;]+);", block)
    return m.group(1).strip().strip("'\"") if m else None


def fetch(url: str, ua: str = UA_TTF) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
        return r.read()


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", s)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("families", nargs="*", default=None, help='e.g. "Inter:wght@400;700"')
    ap.add_argument("--out", default=FONT_DIR)
    a = ap.parse_args()
    families = a.families or DEFAULT_FAMILIES

    os.makedirs(a.out, exist_ok=True)
    query = "&".join(f"family={urllib.parse.quote(f, safe=':;@,')}" for f in families)
    css = fetch(f"https://fonts.googleapis.com/css2?{query}&display=swap").decode("utf-8")

    seen: set[str] = set()
    n = 0
    for block in FACE_RE.findall(css):
        fam, weight, style = _prop(block, "font-family"), _prop(block, "font-weight"), _prop(block, "font-style")
        m = URL_RE.search(block)
        if not (fam and m):
            continue
        url = m.group(1).strip("'\"")
        if not url.endswith(".ttf"):
            print(f"[fonts] skipping non-TTF for {fam} {weight}: {url.rsplit('.', 1)[-1]}")
            continue
        name = f"{slug(fam)}-{weight or '400'}{'-italic' if style == 'italic' else ''}.ttf"
        dest = os.path.join(a.out, name)
        if name in seen:          # the API repeats faces per unicode subset
            continue
        seen.add(name)
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            print(f"[fonts] {name}: present")
            continue
        data = fetch(url)
        with open(dest, "wb") as fh:
            fh.write(data)
        print(f"[fonts] {name}: {len(data) / 1024:.0f} kB")
        n += 1

    print(f"[fonts] {n} new file(s) in {a.out}")
    if not seen:
        print("[fonts] WARNING: nothing downloaded -- the API may have changed its UA sniffing")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
