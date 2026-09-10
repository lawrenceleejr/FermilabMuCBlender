#!/usr/bin/env python3
"""Compose a professional annotation layer over a Cycles render.

Why 2D rather than Blender's compositor: the compositor has no text node, so
"labels in Blender" means 3D text objects, which get lit, fogged and
perspective-warped, and give almost no typographic control. Studios instead
keep the render as a beauty pass and annotate it in 2D. The projection comes
from the render camera itself (`build_scene.py --annotations`), so a label can
never drift off its subject, and everything is sized relative to the image
width, so the same layout holds from a 640 px preview to a 4K still.

Outputs a PNG for slides and a vector PDF/SVG for print and for editing in
Illustrator or Inkscape.

Usage:
    python3 tools/annotate.py renders/fermilab_site_overview.png \\
        --anno out/overview_anno.json --layout overview
    python3 tools/annotate.py ... --debug     # mark raw anchors, to tune a layout
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.font_manager as fm  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FONT_DIR = os.path.join(ROOT, "assets", "fonts")

# --- palette ---------------------------------------------------------------- #
# Three accents carrying one distinction -- proposed / existing / the site
# itself -- and echoing the emitters in the render so a label reads as
# belonging to its subject. Hierarchy comes from size, weight and space, so the
# figure still reads in greyscale or to a colour-blind viewer; colour only
# confirms which of the three a label names.
INK = "#F2F0EB"
INK_DIM = "#9AA3AF"
ACCENT = {
    "collider": "#7FD2FF",     # proposed: collider, RCS, cooling, proton driver
    "tevatron": "#F2B173",     # existing: Tevatron, Main Injector
    "boundary": "#F5F2EC",     # the site itself
    "offsite": "#9E97D6",      # proposed but larger than the campus
    "neutral": "#E9E7E2",      # anything else, e.g. Wilson Hall
}

# --- type scale (1.25 ratio from an 11 pt base, at 1600 px wide) ------------- #
SCALE_BASE = 1600.0
TYPE = {
    "title": 26.0,
    "deck": 12.0,
    "label": 11.5,
    "metric": 9.5,
    "eyebrow": 9.0,
    "meta": 9.0,
    "scale": 9.5,
    "credit": 8.0,
}

# --- per-camera label placement ---------------------------------------------- #
# lx/ly are the label's own position in axes fractions (y up), not an offset, so
# a column of labels can be aligned exactly; `ha` is the side the leader meets.
# Leaders run anchor -> knee -> label as a two-segment elbow, the usual
# architectural callout. Labels sit in two columns clear of the title block and
# the footer, which is why the numbers look regular rather than tuned per label.
# Callouts are given only as an order down each column; positions come from
# LABEL_BAND, so the ladder is always evenly spaced and cannot collide with the
# title or legend bands. Order each column by its anchors' heights and the
# dog-legs run roughly parallel instead of crossing.
LAYOUTS = {
    "overview": {
        "left": ["rcs4", "rcs3", "collider", "cooling", "proton_driver", "boundary"],
        "right": ["tevatron", "detector_a", "wilson", "rcs12", "main_injector", "detector_b"],
    },
    "northeast": {
        "left": ["collider"],
        "right": ["wilson"],
    },
}


def ladder(n):
    """n evenly spaced label baselines down LABEL_BAND, top first."""
    lo, hi = LABEL_BAND
    if n <= 1:
        return [(lo + hi) / 2]
    step = (hi - lo) / (n - 1)
    return [hi - i * step for i in range(n)]


TITLE = dict(
    title="A Future Muon Collider at Fermilab",
    deck="The accelerator chain sited on the existing campus at Batavia, Illinois.",
)

# The footer now carries only what the image cannot say for itself: a credit
# line the data licences ask for. The instant, solar geometry and lens that
# used to sit here were detail a site plan's reader does not need, and they
# crowded the frame.
CREDIT = "Procedural Cycles render \u00b7 star field NASA/GSFC SVS Deep Star Maps 2020, Gaia DR2 (ESA/Gaia/DPAC)"

# One margin and one set of horizontal bands, so every block aligns to the same
# edges and the label spacing is computed rather than typed. Reserving bands
# up front is what stops callouts drifting into the title or the legend.
MARGIN = 0.042
FOOTER_RULE = 0.072          # hairline above the credit line
LEGEND_BAND = (0.105, 0.225)  # scale bar and north needle, right side
LABEL_BAND = (0.265, 0.775)   # callout ladder, both columns
TITLE_FLOOR = 0.800          # nothing else goes above this on the left
COL_X = (MARGIN + 0.098, 1.0 - MARGIN - 0.098)   # the two callout columns


# --------------------------------------------------------------------------- #
# fonts
# --------------------------------------------------------------------------- #
def load_fonts() -> dict[str, fm.FontProperties]:
    """Register the self-hosted IBM Plex TTFs and return FontProperties by role.

    One family throughout -- IBM Plex Sans, which was designed for technical
    contexts and holds up at label sizes. Hierarchy comes from weight and size
    alone, which is steadier than mixing in a second face.
    """
    faces = {
        "sans_light": "IBMPlexSans-300.ttf",
        "sans": "IBMPlexSans-400.ttf",
        "sans_med": "IBMPlexSans-500.ttf",
        "sans_semi": "IBMPlexSans-600.ttf",
    }
    out: dict[str, fm.FontProperties] = {}
    missing = []
    for role, fn in faces.items():
        path = os.path.join(FONT_DIR, fn)
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            out[role] = fm.FontProperties(fname=path)
        else:
            missing.append(fn)
            out[role] = fm.FontProperties(family="DejaVu Sans")
    if missing:
        print(f"[annotate] WARNING: missing fonts {missing}; run tools/get_fonts.py. Falling back to DejaVu.")
    return out


def spaced(text: str, wide: bool = False) -> str:
    """Letter-space an all-caps label, which needs air to stay legible.

    matplotlib exposes no tracking control, so insert the space instead: a thin
    space (U+2009) by default, a full space when `wide`.
    """
    return (" " if not wide else " ").join(text)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def scrim(ax, x0, y0, x1, y1, *, strength=0.62, direction="left", zorder=2):
    """A soft directional gradient behind a text block.

    Type over a photographic background needs separation. A gradient scrim keeps
    the image visible and reads as intentional, where a hard box or a per-letter
    outline reads as a screenshot.
    """
    n = 256
    ramp = np.linspace(0.0, 1.0, n)          # imshow origin="lower": row 0 is the bottom
    if direction == "left":                  # darkest at the left edge
        a = (1.0 - ramp)[None, :].repeat(n, 0)
    elif direction == "right":               # darkest at the right edge
        a = ramp[None, :].repeat(n, 0)
    elif direction == "up":                  # darkest at the top edge
        a = ramp[:, None].repeat(n, 1)
    elif direction == "down":                # darkest at the bottom edge
        a = (1.0 - ramp)[:, None].repeat(n, 1)
    else:
        raise ValueError(f"scrim direction must be left/right/up/down, got {direction!r}")
    rgba = np.zeros((n, n, 4))
    rgba[..., 3] = a * strength
    ax.imshow(rgba, extent=(x0, x1, y0, y1), transform=ax.transAxes,
              origin="lower", aspect="auto", zorder=zorder, interpolation="bilinear")


def leader(ax, ax_x, ax_y, lx, ly, colour, s, width, height, *, ha="left", zorder=6):
    """Dog-leg leader: a 45-degree kick off the anchor, then a horizontal run
    into the text.

    The angled leg is held at a true 45 degrees *on screen* -- computed through
    pixels, since the frame is not square -- so every callout kinks at the same
    angle and the set reads as one drawing rather than ten separate arrows. The
    horizontal run is what carries the eye into the words.
    """
    towards = 1.0 if ha == "left" else -1.0     # which way the label lies
    # horizontal axes-distance that makes the diagonal 45 deg on screen
    run = abs((ly - ax_y) * height) / width
    knee_x = ax_x + towards * run
    # do not overshoot the label; if the anchor is already past it, go straight
    if (towards > 0 and knee_x > lx) or (towards < 0 and knee_x < lx):
        knee_x = lx
    ax.plot([ax_x, knee_x, lx], [ax_y, ly, ly], transform=ax.transAxes, color=colour,
            alpha=0.9, lw=1.0 * s, solid_capstyle="round", solid_joinstyle="miter",
            zorder=zorder,
            path_effects=[patheffects.withStroke(linewidth=2.6 * s, foreground="#05070B99")])
    # a filled dot inside a wide soft ring: reads as a survey mark, and stays
    # legible whether it lands on dark ground or on a bright beamline
    ax.plot([ax_x], [ax_y], transform=ax.transAxes, marker="o", ms=3.2 * s,
            mfc=colour, mec="none", zorder=zorder + 1)
    ax.plot([ax_x], [ax_y], transform=ax.transAxes, marker="o", ms=9.0 * s,
            mfc="none", mec=colour, mew=0.7 * s, alpha=0.45, zorder=zorder)


DRAWN: list[tuple[str, object]] = []      # (role, artist) for the collision checker


def text(ax, x, y, body, fp, size, colour, s, *, ha="left", va="center", zorder=8,
         shadow=True, alpha=1.0, role="text"):
    t = ax.text(x, y, body, transform=ax.transAxes, ha=ha, va=va,
                fontproperties=fp, fontsize=size * s, color=colour, zorder=zorder, alpha=alpha)
    DRAWN.append((role, t))
    if shadow:
        t.set_path_effects([patheffects.withStroke(linewidth=2.2 * s, foreground="#05070Bcc")])
    return t


# --------------------------------------------------------------------------- #
# blocks
# --------------------------------------------------------------------------- #
def axes_dir(vec_px, width, height):
    """Image-space pixel delta (y down) -> axes-fraction delta (y up).

    The frame is not square, so an angle in axes coordinates is not the angle on
    screen; converting through pixels is the only way a bearing stays true.
    """
    dx_px, dy_px = vec_px
    # y is negated: the dump measures downward, matplotlib axes measure upward
    return dx_px / width, -dy_px / height


def draw_title(ax, F, s, copy=None):
    """Title block. No institutional eyebrow -- it implied an official document,
    and dropping it lets the title hold the top of the frame on its own."""
    copy = copy or TITLE
    scrim(ax, 0.0, 0.70, 0.66, 1.0, strength=0.72, direction="left")
    x = MARGIN
    text(ax, x, 0.935, copy["title"], F["sans_light"], TYPE["title"], INK, s, va="top", role="title")
    ax.plot([x, x + 0.062], [0.858, 0.858], transform=ax.transAxes, color=ACCENT["collider"],
            lw=1.8 * s, solid_capstyle="butt", zorder=8)
    text(ax, x, 0.835, copy["deck"], F["sans"], TYPE["deck"], INK_DIM, s, va="top", role="deck")


def draw_footer(ax, F, s, anno):
    """One hairline and one credit line; the scale and north sit just above it."""
    ax.plot([MARGIN, 1.0 - MARGIN], [FOOTER_RULE, FOOTER_RULE], transform=ax.transAxes,
            color=INK, lw=0.7 * s, alpha=0.20, zorder=7)
    text(ax, MARGIN, FOOTER_RULE - 0.030, CREDIT, F["sans"], TYPE["credit"], INK_DIM, s,
         va="center", alpha=0.55, shadow=False, role="credit")


def draw_scale_and_north(ax, F, s, anno, width, height):
    """Graphic scale and north needle, both from the camera's own projection.

    The bar's length is the true projected length of 2 km on the ground at the
    site centre, but it is drawn horizontally: in a legend the length is the
    information and the bearing is not, whereas a tilted bar in a flat footer
    band just reads as a mistake. A perspective view has no single scale, so
    the caption says where this one was measured -- an unqualified bar would be
    a quiet lie, since the same 2 km is shorter at the far boundary.

    The needle does carry a bearing, so it keeps the true projected one. Here it
    points down and to the right because the camera looks south-south-west, so
    a ground vector pointing north comes toward the viewer.
    """
    sc = anno.get("scale", {})
    px_per_km = sc.get("px_per_km_at_reference", 0.0)
    if px_per_km <= 0:
        return
    km = 2.0
    length = km * px_per_km / width
    x0, y = 0.660, LEGEND_BAND[0] + 0.022

    ax.plot([x0, x0 + length], [y, y], transform=ax.transAxes, color=INK, lw=1.9 * s,
            solid_capstyle="butt", zorder=8)
    for f in (0.0, 0.5, 1.0):
        tx = x0 + length * f
        ax.plot([tx, tx], [y, y + 0.016], transform=ax.transAxes, color=INK, lw=1.5 * s, zorder=8)
        text(ax, tx, y + 0.034, f"{km * f:.0f}", F["sans"], TYPE["meta"], INK, s,
             ha="center", va="center", shadow=False, role="legend")
    text(ax, x0 + length + 0.012, y + 0.034, "km", F["sans"], TYPE["meta"], INK, s,
         ha="left", va="center", shadow=False, role="legend")
    text(ax, x0, y - 0.026, "at site centre", F["sans"], TYPE["metric"], INK_DIM, s,
         ha="left", va="center", shadow=False, role="legend")

    # --- north needle, on the true projected bearing --------------------------
    nnx, nny = axes_dir(anno.get("north", {}).get("north_px_per_km", [0.0, -px_per_km]), width, height)
    ln = (nnx ** 2 + nny ** 2) ** 0.5 or 1.0
    ux, uy = nnx / ln * 0.042, nny / ln * 0.042
    cx, cy = 1.0 - MARGIN - 0.012, LEGEND_BAND[0] + 0.030
    ax.annotate("", xy=(cx + ux, cy + uy), xytext=(cx - ux * 0.7, cy - uy * 0.7),
                xycoords=ax.transAxes, textcoords=ax.transAxes, zorder=8,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4 * s, mutation_scale=12 * s,
                                shrinkA=0, shrinkB=0))
    text(ax, cx + ux * 1.55, cy + uy * 1.55, "N", F["sans_semi"], TYPE["scale"], INK, s,
         ha="center", va="center", shadow=False, role="legend")


def draw_features(ax, F, s, anno, layout_name, width, height):
    layout = LAYOUTS.get(layout_name, {})
    feats = anno.get("features", {})
    placed = []
    for side, keys in (("left", layout.get("left", [])), ("right", layout.get("right", []))):
        keys = [k for k in keys if k in feats and feats[k].get("on_screen")]
        ys = ladder(len(keys))
        lx = COL_X[0] if side == "left" else COL_X[1]
        ha = "right" if side == "left" else "left"
        for key, ly in zip(keys, ys):
            placed.append((key, lx, ly, ha, feats[key].get("depth_m", 0.0)))

    # far to near, so a nearer callout draws over a farther one
    for key, lx, ly, ha, _ in sorted(placed, key=lambda t: -t[4]):
        f = feats[key]
        ax_x, ax_y = f["x"], 1.0 - f["y"]
        colour = ACCENT.get(f.get("accent", "neutral"), ACCENT["neutral"])
        leader(ax, ax_x, ax_y, lx, ly, colour, s, width, height, ha=ha)
        pad = 0.011
        tx = lx + (pad if ha == "left" else -pad)
        text(ax, tx, ly + 0.017, f["label"], F["sans_med"], TYPE["label"], INK, s, ha=ha,
             va="center", role=f"label:{key}")
        if f.get("metric"):
            text(ax, tx, ly - 0.017, f["metric"], F["sans"], TYPE["metric"], colour, s,
                 ha=ha, va="center", alpha=0.95, role=f"metric:{key}")

    wanted = set(layout.get("left", [])) | set(layout.get("right", []))
    missing = [k for k in wanted if k not in feats or not feats[k].get("on_screen")]
    if missing:
        print(f"[annotate] not drawn (off screen or absent): {', '.join(sorted(missing))}")
    return len(placed)


def check_collisions(fig, ax, width, height, *, pad_px=6.0):
    """Measure every text box and report overlaps.

    "Mind the spacing" is only checkable if it is measured, so this renders
    once to get real ink extents from FreeType and then tests every pair,
    plus each label against the title block, the footer rule and the legend.
    A label and its own metric are allowed to sit close; everything else needs
    `pad_px` of clear space.
    """
    # a bare Figure has no renderer until one is attached; Agg gives real
    # FreeType extents, which is the whole point of measuring rather than guessing
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    boxes = []
    for role, t in DRAWN:
        bb = t.get_window_extent(renderer)
        boxes.append((role, bb.x0, bb.y0, bb.x1, bb.y1))

    def overlap(a, b, pad):
        return not (a[3] + pad <= b[1] or b[3] + pad <= a[1]
                    or a[4] + pad <= b[2] or b[4] + pad <= a[2])

    def same_callout(r1, r2):
        return r1.split(":", 1)[-1] == r2.split(":", 1)[-1] and ":" in r1 and ":" in r2

    hits = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            pad = 1.0 if same_callout(a[0], b[0]) else pad_px
            if overlap(a, b, pad):
                hits.append((a[0], b[0], "text/text"))

    # text against the reserved zones, in pixels
    zones = {
        "footer/credit band": (0.0, 0.0, 1.0, FOOTER_RULE),
        "legend band": (0.62, LEGEND_BAND[0] - 0.01, 1.0, LEGEND_BAND[1]),
    }
    for role, x0, y0, x1, y1 in boxes:
        if role in ("credit", "legend"):
            continue
        for zname, (zx0, zy0, zx1, zy1) in zones.items():
            z = (zname, zx0 * width, zy0 * height, zx1 * width, zy1 * height)
            if overlap((role, x0, y0, x1, y1), z, 2.0):
                hits.append((role, zname, "text/zone"))

    # anything running off the canvas
    for role, x0, y0, x1, y1 in boxes:
        if x0 < 2 or y0 < 2 or x1 > width - 2 or y1 > height - 2:
            hits.append((role, "canvas edge", "clipped"))

    if hits:
        print(f"[annotate] {len(hits)} spacing problem(s):")
        for a, b, kind in hits:
            print(f"    {kind:11s} {a}  <->  {b}")
    else:
        print(f"[annotate] spacing OK: {len(boxes)} text blocks, no overlaps within {pad_px:.0f} px")
    return hits


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build(args) -> int:
    img = np.asarray(Image.open(args.image).convert("RGB"))
    height, width = img.shape[:2]
    with open(args.anno) as fh:
        anno = json.load(fh)

    rw = anno.get("render", {}).get("width")
    if rw and abs(rw - width) > 1:
        print(f"[annotate] NOTE: annotations were dumped for {rw} px wide, image is {width} px. "
              "Normalised coordinates still line up; type is scaled to the image.")

    s = width / SCALE_BASE          # keeps type and rules proportional at any size
    dpi = 100.0
    F = load_fonts()

    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.imshow(img, extent=(0, 1, 0, 1), transform=ax.transAxes, aspect="auto", zorder=0,
              interpolation="antialiased")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


    if args.debug:
        for key, f in anno.get("features", {}).items():
            x, y = f["x"], 1.0 - f["y"]
            ax.plot([x], [y], transform=ax.transAxes, marker="+", ms=14 * s, mew=1.4 * s,
                    color="#FF3B30", zorder=9)
            text(ax, x + 0.006, y, f"{key}  ({x:.3f},{f['y']:.3f})", F["sans"], 8.5, "#FF3B30", s)
        for gx in np.arange(0.1, 1.0, 0.1):
            ax.axvline(gx, color="#FF3B3055", lw=0.6 * s, zorder=4)
            ax.axhline(gx, color="#FF3B3055", lw=0.6 * s, zorder=4)
    else:
        draw_title(ax, F, s, copy=dict(
            title=args.title or TITLE["title"],
            deck=(args.deck.replace("\\n", "\n") if args.deck else TITLE["deck"]),
        ))
        draw_footer(ax, F, s, anno)
        draw_scale_and_north(ax, F, s, anno, width, height)
        draw_features(ax, F, s, anno, args.layout, width, height)

    hits = check_collisions(fig, ax, width, height) if not args.debug else []
    if hits and args.strict:
        print("[annotate] --strict: refusing to write a figure with spacing problems")
        return 1

    stem = args.out or os.path.splitext(args.image)[0] + ("_debug" if args.debug else "_annotated")
    written = []
    for ext in (args.formats.split(",") if args.formats else ["png"]):
        ext = ext.strip().lower()
        path = f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi, facecolor="#000000", edgecolor="none",
                    pad_inches=0, bbox_inches=None)
        written.append(f"{path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    print("[annotate] wrote " + "; ".join(written))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", help="the rendered PNG to annotate")
    p.add_argument("--anno", required=True, help="JSON from build_scene.py --annotations")
    p.add_argument("--layout", default="overview", choices=sorted(LAYOUTS))
    p.add_argument("--out", default="", help="output path stem (default: <image>_annotated)")
    p.add_argument("--formats", default="png", help="comma list: png,pdf,svg")
    p.add_argument("--debug", action="store_true", help="mark raw anchors and a decile grid")
    p.add_argument("--strict", action="store_true", help="fail rather than write a figure with text collisions")
    p.add_argument("--title", default="", help="override the title line")
    p.add_argument("--deck", default="", help="override the standfirst; \\n splits lines")
    return build(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
