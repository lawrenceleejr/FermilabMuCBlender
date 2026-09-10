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
# Accent hues echo the emitters in the render so a label reads as belonging to
# its subject. Hierarchy comes from size, weight and space; colour only confirms
# it, so the layout still works for a colour-blind reader or in greyscale.
INK = "#F2F0EB"
INK_DIM = "#9AA3AF"
ACCENT = {
    "collider": "#7FD2FF",
    "tevatron": "#F2B173",
    "beam": "#CFE4FF",
    "boundary": "#F5F2EC",
    "neutral": "#E9E7E2",
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
}

# --- per-camera label placement ---------------------------------------------- #
# lx/ly are the label's own position in axes fractions (y up), not an offset, so
# a column of labels can be aligned exactly; `ha` is the side the leader meets.
# Leaders run anchor -> knee -> label as a two-segment elbow, the usual
# architectural callout. Labels sit in two columns clear of the title block and
# the footer, which is why the numbers look regular rather than tuned per label.
LAYOUTS = {
    "overview": {
        # left column
        "detector_b":    dict(lx=0.150, ly=0.560, ha="right"),
        "collider":      dict(lx=0.150, ly=0.450, ha="right"),
        "boundary":      dict(lx=0.150, ly=0.255, ha="right"),
        # right column
        "linac":         dict(lx=0.850, ly=0.605, ha="left"),
        "wilson":        dict(lx=0.850, ly=0.495, ha="left"),
        "detector_a":    dict(lx=0.850, ly=0.385, ha="left"),
        "tevatron":      dict(lx=0.850, ly=0.275, ha="left"),
        # centre-bottom
        "main_injector": dict(lx=0.430, ly=0.300, ha="right"),
    },
    "northeast": {
        "wilson":        dict(lx=0.820, ly=0.660, ha="left"),
        "collider":      dict(lx=0.170, ly=0.400, ha="right"),
    },
}

TITLE = dict(
    eyebrow="FERMI NATIONAL ACCELERATOR LABORATORY",
    title="A Muon Collider on the Existing Site",
    deck=("A 10 km collider ring sited within the existing campus at Batavia, Illinois,\n"
          "reusing the Tevatron tunnel and the injector chain already in the ground."),
)


# --------------------------------------------------------------------------- #
# fonts
# --------------------------------------------------------------------------- #
def load_fonts() -> dict[str, fm.FontProperties]:
    """Register the self-hosted IBM Plex TTFs and return FontProperties by role.

    IBM Plex is a superfamily, so the sans and mono pair by construction. Sans
    carries the words; mono carries the numbers, where its fixed advance widths
    keep columns of figures aligned and visually signal 'measured data'.
    """
    faces = {
        "sans_light": "IBMPlexSans-300.ttf",
        "sans": "IBMPlexSans-400.ttf",
        "sans_med": "IBMPlexSans-500.ttf",
        "sans_semi": "IBMPlexSans-600.ttf",
        "mono": "IBMPlexMono-400.ttf",
        "mono_med": "IBMPlexMono-500.ttf",
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


def leader(ax, ax_x, ax_y, lx, ly, colour, s, *, ha="left", zorder=6):
    """Two-segment elbow leader from an anchor dot to the label baseline.

    The short horizontal run into the text is what makes a callout look drawn
    rather than dropped: the eye follows the horizontal into the words.
    """
    run = 0.026 if ha == "left" else -0.026
    knee_x = lx + run
    ax.plot([ax_x, knee_x, lx], [ax_y, ly, ly], transform=ax.transAxes, color=colour,
            alpha=0.85, lw=0.9 * s, solid_capstyle="round", solid_joinstyle="round", zorder=zorder)
    # a filled dot with a wide soft ring: reads as a survey mark, and stays
    # visible whether it lands on dark ground or on a bright beamline
    ax.plot([ax_x], [ax_y], transform=ax.transAxes, marker="o", ms=3.2 * s,
            mfc=colour, mec="none", zorder=zorder + 1)
    ax.plot([ax_x], [ax_y], transform=ax.transAxes, marker="o", ms=9.0 * s,
            mfc="none", mec=colour, mew=0.7 * s, alpha=0.40, zorder=zorder)


def text(ax, x, y, body, fp, size, colour, s, *, ha="left", va="center", zorder=8, shadow=True, alpha=1.0):
    t = ax.text(x, y, body, transform=ax.transAxes, ha=ha, va=va,
                fontproperties=fp, fontsize=size * s, color=colour, zorder=zorder, alpha=alpha)
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


def draw_title(ax, F, s):
    scrim(ax, 0.0, 0.58, 0.66, 1.0, strength=0.68, direction="left")
    x = 0.038
    text(ax, x, 0.950, spaced(TITLE["eyebrow"]), F["sans_med"], TYPE["eyebrow"], ACCENT["collider"], s, va="top")
    text(ax, x, 0.905, TITLE["title"], F["sans_light"], TYPE["title"], INK, s, va="top")
    ax.plot([x, x + 0.075], [0.822, 0.822], transform=ax.transAxes, color=ACCENT["collider"],
            lw=1.7 * s, solid_capstyle="butt", zorder=8)
    text(ax, x, 0.797, TITLE["deck"], F["sans"], TYPE["deck"], INK_DIM, s, va="top")


def draw_footer(ax, F, s, anno):
    """A footer band rather than corner blocks, so the image area stays clear.

    The mono line carries the facts that make the picture checkable instead of
    merely decorative: the instant, the solar geometry, the lens.
    """
    sky = anno.get("sky") or {}
    cam = anno.get("camera", {})
    loc = cam.get("location", [0, 0, 0])
    scrim(ax, 0.0, 0.0, 1.0, 0.125, strength=0.74, direction="down")
    ax.plot([0.0, 1.0], [0.118, 0.118], transform=ax.transAxes, color=INK,
            lw=0.7 * s, alpha=0.22, zorder=7)

    text(ax, 0.038, 0.082, "Fermilab — proposed muon collider", F["sans_med"], TYPE["label"], INK, s)
    text(ax, 0.038, 0.053, "Site overview, looking south-south-west", F["sans"], TYPE["metric"], INK_DIM, s)

    sun_a = sky.get("sun_altitude_deg", float("nan"))
    sun_z = sky.get("sun_azimuth_deg", float("nan"))
    gc_a = sky.get("galactic_centre_altitude_deg", float("nan"))
    gc_z = sky.get("galactic_centre_azimuth_deg", float("nan"))
    utc = sky.get("utc", "n/a")
    phase = sky.get("twilight_phase", "")
    col = 0.335
    text(ax, col, 0.082, f"{utc} UTC   ·   {phase}", F["mono_med"], TYPE["meta"], INK, s, alpha=0.92)
    text(ax, col, 0.053,
         f"sun {sun_a:+.1f}° alt {sun_z:.0f}° az   ·   galactic centre {gc_a:.0f}° alt {gc_z:.0f}° az   ·   "
         f"{cam.get('lens_mm', 0):.0f} mm at {loc[2]:.0f} m",
         F["mono"], TYPE["meta"], INK_DIM, s)
    text(ax, 0.038, 0.023, "Cycles / procedural scene · star field NASA GSFC SVS Deep Star Maps 2020, Gaia DR2 (ESA/Gaia/DPAC)",
         F["mono"], TYPE["meta"] * 0.88, INK_DIM, s, alpha=0.6)


def draw_scale_and_north(ax, F, s, anno, width, height):
    """Scale bar and north needle, both from the camera's own projection.

    A perspective view has no single scale, so the bar is measured on the ground
    at the site centre and labelled as such -- an unqualified bar would be a
    quiet lie, since the same 2 km is shorter at the far boundary than the near.
    """
    sc = anno.get("scale", {})
    px_per_km = sc.get("px_per_km_at_reference", 0.0)
    if px_per_km <= 0:
        return
    km = 2.0
    ex, ey = axes_dir(sc.get("east_px_per_km", [px_per_km, 0.0]), width, height)
    # The bar measures 2 km along the ground's east-west line; which end is east
    # does not matter, so always draw it left-to-right, the way a bar is read.
    if ex < 0:
        ex, ey = -ex, -ey
    x0, y0 = 0.575, 0.168
    dx, dy = ex * km, ey * km

    ax.plot([x0, x0 + dx], [y0, y0 + dy], transform=ax.transAxes, color=INK, lw=1.9 * s,
            solid_capstyle="butt", zorder=8, path_effects=[patheffects.withStroke(linewidth=3.6 * s, foreground="#05070Baa")])
    # ticks perpendicular to the bar, at 0, 1 and 2 km
    nx, ny = -dy, dx
    n_len = (nx ** 2 + ny ** 2) ** 0.5
    nx, ny = nx / n_len * 0.011, ny / n_len * 0.011
    for f in (0.0, 0.5, 1.0):
        tx, ty = x0 + dx * f, y0 + dy * f
        ax.plot([tx, tx + nx], [ty, ty + ny], transform=ax.transAxes, color=INK, lw=1.5 * s, zorder=8)
        text(ax, tx + nx * 2.0, ty + ny * 2.0, f"{km * f:.0f}", F["mono"], TYPE["meta"], INK, s,
             ha="center", va="center")
    mid_x, mid_y = x0 + dx * 0.5, y0 + dy * 0.5
    text(ax, mid_x - nx * 2.4, mid_y - ny * 2.4, "km at site centre", F["sans"], TYPE["metric"],
         INK_DIM, s, ha="center", va="center", alpha=0.9)

    # --- north needle -------------------------------------------------------------
    nnx, nny = axes_dir(anno.get("north", {}).get("north_px_per_km", [0.0, -px_per_km]), width, height)
    ln = (nnx ** 2 + nny ** 2) ** 0.5 or 1.0
    ux, uy = nnx / ln * 0.055, nny / ln * 0.055
    cx, cy = 0.868, 0.215
    ax.annotate("", xy=(cx + ux * 0.6, cy + uy * 0.6), xytext=(cx - ux * 0.6, cy - uy * 0.6),
                xycoords=ax.transAxes, textcoords=ax.transAxes, zorder=8,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4 * s, mutation_scale=13 * s,
                                shrinkA=0, shrinkB=0))
    text(ax, cx + ux * 1.05, cy + uy * 1.05, "N", F["sans_semi"], TYPE["scale"], INK, s,
         ha="center", va="center")


def draw_features(ax, F, s, anno, layout_name, width, height):
    layout = LAYOUTS.get(layout_name, {})
    feats = anno.get("features", {})
    # far to near, so a nearer label draws over a farther one rather than under
    order = sorted(feats.items(), key=lambda kv: -kv[1].get("depth_m", 0.0))
    drawn = 0
    for key, f in order:
        L = layout.get(key)
        if L is None or not f.get("on_screen", False):
            continue
        ax_x, ax_y = f["x"], 1.0 - f["y"]          # axes fractions have y up
        lx, ly, ha = L["lx"], L["ly"], L.get("ha", "left")
        colour = ACCENT.get(f.get("accent", "neutral"), ACCENT["neutral"])
        leader(ax, ax_x, ax_y, lx, ly, colour, s, ha=ha)
        pad = 0.011
        tx = lx + (pad if ha == "left" else -pad)
        text(ax, tx, ly + 0.017, f["label"], F["sans_med"], TYPE["label"], INK, s, ha=ha, va="center")
        if f.get("metric"):
            text(ax, tx, ly - 0.017, f["metric"], F["mono"], TYPE["metric"], colour, s,
                 ha=ha, va="center", alpha=0.95)
        drawn += 1
    skipped = [k for k in layout if k not in feats or not feats[k].get("on_screen")]
    if skipped:
        print(f"[annotate] not drawn (off screen or absent from the dump): {', '.join(skipped)}")
    return drawn


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
            text(ax, x + 0.006, y, f"{key}  ({x:.3f},{f['y']:.3f})", F["mono"], 8.5, "#FF3B30", s)
        for gx in np.arange(0.1, 1.0, 0.1):
            ax.axvline(gx, color="#FF3B3055", lw=0.6 * s, zorder=4)
            ax.axhline(gx, color="#FF3B3055", lw=0.6 * s, zorder=4)
    else:
        draw_title(ax, F, s)
        draw_footer(ax, F, s, anno)
        draw_scale_and_north(ax, F, s, anno, width, height)
        draw_features(ax, F, s, anno, args.layout, width, height)

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
    return build(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
