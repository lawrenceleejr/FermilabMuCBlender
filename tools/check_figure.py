#!/usr/bin/env python3
"""Measure a finished figure against the things design critique actually caught.

`annotate.py --strict` checks the layout it drew: overlaps, margins, leader
crossings. It cannot check the result, because the result depends on the render
underneath -- whether the type has a ground to sit on, whether the scrim has a
visible edge, whether the subject carries more tonal mass than the sky. Those
were the findings that mattered most and none of them is a property of the
layout alone, so they get measured here, on pixels.

What it reports, and why each one is here:

  figure/ground   Band means for the sky, the subject and the footer. A subject
                  at 11 % of frame area under a bright sky cannot be the focal
                  point however many arrows point at it.
  scrim edges     The largest single-row and single-column luminance step in
                  the frame interior. A gradient with one hard side reads as a
                  compositing bug; the whole justification for a scrim is that
                  it has no locatable edge.
  contrast        WCAG ratio for each text block against the median luminance
                  actually behind it. 4.5 is the AA floor for normal text.
  clipping        Fraction of pixels at 0 and at 255.

Usage:
    python3 tools/check_figure.py out/anno_v9.png [--anno out/overview_anno.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def rel_lum(rgb):
    """WCAG relative luminance from 0-255 sRGB."""
    c = np.asarray(rgb, float) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]


def ratio(l1, l2):
    a, b = max(l1, l2), min(l1, l2)
    return (a + 0.05) / (b + 0.05)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--anno", default="", help="annotation JSON, to locate the subject")
    ap.add_argument("--base", default="", help="the un-annotated render, to isolate the overlay")
    ap.add_argument("--blocks", default="", help="JSON of text blocks from annotate.py --dump-blocks")
    a = ap.parse_args()

    img = np.asarray(Image.open(a.image).convert("RGB")).astype(float)
    h, w = img.shape[:2]
    L = np.asarray(Image.open(a.image).convert("L")).astype(float)
    print(f"[check] {a.image}  {w}x{h}")

    # --- figure / ground -----------------------------------------------------
    sky = L[:int(h * 0.13)]
    mid = L[int(h * 0.30):int(h * 0.78)]
    # the same band restricted to the picture columns. The callout gutters are
    # annotation, not subject: bright label type inflates the full-width figure
    # and a scrim behind it deflates the same number, so both are reported and
    # neither is the whole answer.
    mid_pic = L[int(h * 0.30):int(h * 0.78), int(w * 0.28):int(w * 0.72)]
    foot = L[int(h * 0.85):]
    print("\nfigure/ground (mean luminance 0-255)")
    print(f"  sky band      rows 0-{int(h * .13):<5d} {sky.mean():6.1f}")
    print(f"  subject band  rows {int(h * .30)}-{int(h * .78):<5d} {mid.mean():6.1f} "
          f"(picture columns only: {mid_pic.mean():.1f})")
    print(f"  footer band   rows {int(h * .85)}-{h:<5d} {foot.mean():6.1f}")
    r_all = mid.mean() / max(sky.mean(), 1e-6)
    r_pic = mid_pic.mean() / max(sky.mean(), 1e-6)
    verdict = "subject leads" if r_pic >= 0.72 else "SKY DOMINATES the subject"
    print(f"  -> {verdict} (subject/sky {r_all:.2f} full width, {r_pic:.2f} picture columns)")

    # brightest cell of a coarse grid: where does the eye actually go
    gy, gx = 5, 8
    cells = L[:gy * (h // gy), :gx * (w // gx)].reshape(gy, h // gy, gx, w // gx).mean((1, 3))
    j, i = np.unravel_index(cells.argmax(), cells.shape)
    print(f"  brightest cell of a {gy}x{gx} grid: row {j} of {gy}, col {i} of {gx} "
          f"({'sky' if j == 0 else 'below the horizon'})")

    # --- scrim / seam detection ---------------------------------------------
    # Three traps, all hit on the way here. Measured over the full width, one
    # 10 pt credit line moves the row mean by tens of luminance units and
    # swamps what is being looked for. A single-line difference spikes on any
    # thin bright feature, so a 1 px emissive beamline reads as a seam. And
    # after fixing both, the largest sustained step in the frame is the
    # *horizon* -- a real feature that no amount of profile filtering will
    # distinguish from a scrim edge.
    #
    # So the measurement is made on the difference against the base render.
    # The scrim is an alpha overlay: differencing cancels the horizon, the
    # beamlines and the terrain, and leaves exactly what the annotation layer
    # put there. Without --base the seam check is skipped rather than guessed.
    def worst_step(profile, lo, win=8, gap=3):
        """Largest level change across a window, in luminance units.

        `win` and `gap` arrive already scaled to the figure -- see the caller.
        Given absolute pixel counts they measure a different fraction of the
        image at every resolution, which is how a perfectly smooth 30 px fade
        came to be reported as a 10 L step at 800 px and nothing at 1600 px.
        """
        win = max(2, win)
        gap = max(1, gap)
        best = (0.0, lo)
        for i in range(win + gap, len(profile) - win - gap):
            d = abs(np.median(profile[i + gap:i + gap + win])
                    - np.median(profile[i - gap - win:i - gap]))
            if d > best[0]:
                best = (float(d), i + lo)
        return best

    if a.base and os.path.exists(a.base):
        base = np.asarray(Image.open(a.base).convert("L").resize((w, h))).astype(float)
        d = L - base
        # Median along each line, not mean: type occupies a small fraction of
        # any row or column, so the median mostly ignores it while still
        # tracking the overlay's level. A mean kept measuring whichever text
        # block fell in the window.
        #
        # Two residual traps, both of which produced a confident wrong answer.
        # The median is *not* robust to a block wider than half the frame, and
        # the 36 pt title is: its rows read as a 7 L step. So the row scan stays
        # below the title block. And on 8-bit medians the profile jitters by a
        # unit or two, which a raw window comparison reports as a 6 L step in a
        # perfectly smooth ramp -- so the profile is smoothed first.
        # Every window below is a fraction of the frame, not a pixel count.
        # The numbers were tuned on a 1600 px wide figure and the layout they
        # measure scales with width, so they scale with it too.
        fs = w / 1600.0
        win, gap = int(round(8 * fs)), int(round(3 * fs))
        k = max(3, int(round(9 * fs)) | 1)

        def smooth(v, k=k):
            if len(v) < k:
                return v
            return np.convolve(v, np.ones(k) / k, mode="valid")

        # The scan covers the picture region only: below the title block and
        # above the legend band. Both of those are dense with type, and type
        # wider than half the frame defeats a median.
        r0, r1 = int(h * 0.24), int(h * 0.76)
        rows = smooth(np.median(d[r0:r1, :], axis=1))
        c0, c1 = int(w * 0.03), int(w * 0.97)
        cols = smooth(np.median(d[r0:r1, c0:c1], axis=0))
        r0 += k // 2                                # the valid-mode convolution offset
        c0 += k // 2
        r_step, r_at = worst_step(rows, r0, win=win, gap=gap)
        c_step, c_at = worst_step(cols, c0, win=win, gap=gap)
        print("\nscrim edges (sustained step in annotated minus base, clear of type)")
        print(f"  row shift  {r_step:5.2f} L at row {r_at} ({r_at / h:.2f} of height)")
        print(f"  col shift  {c_step:5.2f} L at col {c_at} ({c_at / w:.2f} of width)")
        # Threshold set from this tool's own measured noise floor, not chosen to
        # make the current figure pass. In a window holding both callout type
        # and a gradient the reading sits at 5-6 L: the median is only partly
        # robust to type, 8-bit profiles jitter by a unit or two, and where a
        # feathered ramp meets its plateau there is a real slope kink (which is
        # continuous in level, so not a seam). The hard-edged scrim this
        # replaced measured 13-19 L on the same test, so 8 L discriminates the
        # defect from the floor with room either side.
        seams = [(n, v, at) for n, v, at in
                 (("row", r_step, r_at), ("col", c_step, c_at)) if v > 8.0]
        for label, v, at in seams:
            print(f"  -> VISIBLE {label} seam at {at}: the overlay changes level by "
                  f"{v:.0f} L across a few pixels, which reads as a pasted rectangle")
        if not seams:
            print("  -> no locatable scrim edge: the overlay has no hard side")
    else:
        print("\nscrim edges: skipped (pass --base <render> to measure the overlay alone)")

    # --- text contrast, if the block boxes were dumped ----------------------
    if a.blocks and os.path.exists(a.blocks):
        blocks = json.load(open(a.blocks))
        # WCAG 2.1 SC 1.4.3 sets two floors, not one: 4.5:1 for body text and
        # 3:1 for large-scale text, which it defines as 18 pt or 14 pt bold. A
        # single 4.5 floor was holding a 36.5 pt title to the body-text
        # standard. The size compared is the design size, since the output
        # resolution does not change whether a title is display type.
        def floor_for(b):
            size = float(b.get("size", 0.0))
            weight = str(b.get("weight", "normal"))
            bold = weight in ("bold", "semibold", "600", "700", "800", "900")
            return 3.0 if (size >= 18.0 or (bold and size >= 14.0)) else 4.5

        print("\ntext contrast (WCAG, against the median luminance behind each block)")
        print("  floor is 4.5, or 3.0 for large-scale text (18 pt, or 14 pt bold)")
        worst = []
        for b in blocks:
            x0, y0, x1, y1 = (int(b["x0"]), int(b["y0"]), int(b["x1"]), int(b["y1"]))
            y0i, y1i = h - y1, h - y0                 # matplotlib y-up -> image rows
            patch = img[max(y0i, 0):min(y1i, h), max(x0, 0):min(x1, w)]
            if patch.size == 0:
                continue
            bg = float(np.median(rel_lum(patch).ravel()))
            fg = float(rel_lum(np.array(b["rgb"])))
            cr = ratio(fg, bg)
            worst.append((cr, b["role"], floor_for(b)))
        for cr, role, fl in sorted(worst, key=lambda t: t[0] / t[2])[:8]:
            flag = f"  FAILS AA (floor {fl:g})" if cr < fl else ""
            print(f"  {cr:6.2f}  {role}{flag}")
        n_fail = sum(1 for cr, _, fl in worst if cr < fl)
        print(f"  -> {len(worst)} blocks, {n_fail} below their AA floor")

    # --- subject extent, if the annotations are available -------------------
    if a.anno and os.path.exists(a.anno):
        anno = json.load(open(a.anno))
        f = anno.get("features", {})
        xs = [v["x"] for v in f.values() if v.get("on_screen")]
        ys = [v["y"] for v in f.values() if v.get("on_screen")]
        if xs:
            # the spread of the *labelled* anchors, which is narrower than the
            # site: the boundary contributes one point, not its whole outline
            print(f"\nlabelled-anchor spread (not the site's extent)")
            print(f"  x {min(xs):.3f}..{max(xs):.3f} = {(max(xs) - min(xs)) * 100:.0f}% of width")
            print(f"  y {min(ys):.3f}..{max(ys):.3f} = {(max(ys) - min(ys)) * 100:.0f}% of height")
        sc = anno.get("scale", {})
        npk = anno.get("north", {}).get("north_px_per_km")
        if sc.get("px_per_km_at_reference") and npk:
            e = sc["px_per_km_at_reference"]
            nn = (npk[0] ** 2 + npk[1] ** 2) ** 0.5
            print(f"  projection anisotropy east/north {e / max(nn, 1e-6):.2f} : 1 "
                  f"(the graphic scale must show both legs)")

    # --- clipping ------------------------------------------------------------
    lo = float((L <= 0.5).mean() * 100)
    hi = float((L >= 254.5).mean() * 100)
    print(f"\nclipping: {lo:.2f}% at black, {hi:.2f}% at white")
    return 0


if __name__ == "__main__":
    sys.exit(main())
