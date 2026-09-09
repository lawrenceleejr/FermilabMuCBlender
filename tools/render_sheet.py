#!/usr/bin/env python3
"""Inspection sheet for a render: the image, 3 zoomed crops (centre, hero,
corner) and a luminance histogram with clipping statistics. Tufte-plain:
no chart junk, just the data.

    python3 tools/render_sheet.py out/preview.png [--hero 0.42,0.45] [--out sheet.png]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
from PIL import Image, ImageDraw


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--hero", default="0.45,0.45", help="normalised x,y of the hero crop centre")
    p.add_argument("--out", default="")
    p.add_argument("--width", type=int, default=1400)
    a = p.parse_args()

    im = Image.open(a.image).convert("RGB")
    W, H = im.size
    arr = np.asarray(im).astype(np.float32) / 255.0
    lum = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    hx, hy = (float(v) for v in a.hero.split(","))

    # stats
    clip_hi = float((lum > 0.98).mean() * 100)
    clip_lo = float((lum < 0.02).mean() * 100)
    mean = float(lum.mean())
    p50, p90, p99 = (float(np.percentile(lum, q)) for q in (50, 90, 99))
    print(f"{a.image}: {W}x{H}  mean={mean:.3f}  p50={p50:.3f} p90={p90:.3f} p99={p99:.3f}  clipped: {clip_hi:.2f}% white, {clip_lo:.2f}% black")

    # layout: image on top, crops + histogram below
    main_w = a.width
    main_h = int(H * main_w / W)
    crop_size = main_w // 3
    sheet = Image.new("RGB", (main_w, main_h + crop_size + 120), (12, 12, 12))
    sheet.paste(im.resize((main_w, main_h), Image.LANCZOS), (0, 0))

    def crop(cx, cy, frac=0.18):
        cw, ch = int(W * frac), int(W * frac)
        x0 = int(max(0, min(W - cw, cx * W - cw / 2)))
        y0 = int(max(0, min(H - ch, cy * H - ch / 2)))
        return im.crop((x0, y0, x0 + cw, y0 + ch)).resize((crop_size, crop_size), Image.LANCZOS)

    for i, (cx, cy) in enumerate(((hx, hy), (0.5, 0.5), (0.12, 0.85))):
        sheet.paste(crop(cx, cy), (i * crop_size, main_h))

    d = ImageDraw.Draw(sheet)
    hist, _ = np.histogram(lum, bins=128, range=(0, 1))
    hist = hist / hist.max()
    hy0 = main_h + crop_size + 110
    for i, v in enumerate(hist):
        x = int(i * main_w / 128)
        d.line([(x, hy0), (x, hy0 - int(v * 95))], fill=(200, 200, 200), width=max(1, main_w // 128 - 1))
    d.text((6, main_h + crop_size + 4), f"mean {mean:.3f}  p50 {p50:.3f}  p90 {p90:.3f}  p99 {p99:.3f}   clipped white {clip_hi:.2f}%  black {clip_lo:.2f}%", fill=(230, 230, 230))
    out = a.out or a.image.rsplit(".", 1)[0] + "_sheet.png"
    sheet.save(out)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
