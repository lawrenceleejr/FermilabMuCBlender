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
# dx/dy are offsets from the projected anchor in fractions of image width, and
# `ha` is which side of the label the leader meets. Leaders run anchor -> knee
# -> label as a two-segment elbow, the usual architectural callout. Tuned by eye
# against --debug so leaders neither cross each other nor cover their subjects.
LAYOUTS = {
    "overview": {
        "boundary":      dict(dx= 0.150, dy=-0.120, ha="left"),
        "collider":      dict(dx=-0.230, dy=-0.105, ha="right"),
        "detector_a":    dict(dx=-0.120, dy= 0.115, ha="right"),
        "detector_b":    dict(dx=-0.155, dy=-0.055, ha="right"),
        "linac":         dict(dx=-0.060, dy=-0.150, ha="right"),
        "tevatron":      dict(dx= 0.175, dy= 0.070, ha="left"),
        "main_injector": dict(dx= 0.040, dy= 0.150, ha="left"),
        "wilson":        dict(dx= 0.135, dy=-0.035, ha="left"),
    },
    "northeast": {
        "wilson":        dict(dx= 0.140, dy=-0.150, ha="left"),
        "collider":      dict(dx=-0.150, dy=-0.100, ha="right"),
    },
}

TITLE = dict(
    eyebrow="FERMI NATIONAL ACCELERATOR LABORATORY",
    title="A Muon Collider on the Existing Site",
    deck=("The proposed 10 km collider ring sited inside Fermilab's 27 km² campus at Batavia, Illinois,\n"
          "reusing the Tevatron tunnel and the existing injector chain."),
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


def spaced(text: str, em: float = 0.18) -> str:
    """Letter-spacing for small caps labels, which need air to stay legible.

    matplotlib has no tracking control, so widen with thin spaces.
    """
    gap = " " if em < 0.25 else " "
    return gap.join(text)


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
    ramp = np.linspace(0.0, 1.0, n)
    if direction == "left":
        a = (1.0 - ramp)[None, :].repeat(n, 0)
    elif direction == "right":
        a = ramp[None, :].repeat(n, 0)
    elif direction == "bottom":
        a = ramp[:, None].repeat(n, 1)
    else:
        a = (1.0 - ramp)[:, None].repeat(n, 1)
    rgba = np.zeros((n, n, 4))
    rgba[..., 3] = a * strength
    ax.imshow(rgba, extent=(x0, x1, y0, y1), transform=ax.transAxes,
              origin="lower", aspect="auto", zorder=zorder, interpolation="bilinear")


def leader(ax, ax_x, ax_y, lx, ly, colour, s, *, ha="left", zorder=6):
    """Two-segment elbow leader from an anchor dot to a label baseline."""
    knee_x = lx - (0.022 * s / s) if ha == "left" else lx + 0.022
    knee_x = lx - 0.022 if ha == "left" else lx + 0.022
    pts_x = [ax_x, knee_x, lx]
    pts_y = [ax_y, ly, ly]
    ax.plot(pts_x, pts_y, transform=ax.transAxes, color=colour, alpha=0.8,
            lw=0.9 * s, solid_capstyle="round", zorder=zorder)
    ax.plot([ax_x], [ax_y], transform=ax.transAxes, marker="o", ms=3.4 * s,
            mfc=colour, mec="none", zorder=zorder + 1)
    ax.plot([ax_x], [ax_y], transform=ax.transAxes, marker="o", ms=8.0 * s,
            mfc="none", mec=colour, mew=0.7 * s, alpha=0.45, zorder=zorder)


def text(ax, x, y, body, fp, size, colour, s, *, ha="left", va="center", zorder=8, shadow=True, alpha=1.0):
    t = ax.text(x, y, body, transform=ax.transAxes, ha=ha, va=va,
                fontproperties=fp, fontsize=size * s, color=colour, zorder=zorder, alpha=alpha)
    if shadow:
        t.set_path_effects([patheffects.withStroke(linewidth=2.2 * s, foreground="#05070Bcc")])
    return t


# --------------------------------------------------------------------------- #
# blocks
# --------------------------------------------------------------------------- #
def draw_title(ax, F, s, meta):
    scrim(ax, 0.0, 0.60, 0.62, 1.0, strength=0.66, direction="left")
    x = 0.038
    text(ax, x, 0.945, spaced(TITLE["eyebrow"]), F["sans_med"], TYPE["eyebrow"], ACCENT["collider"], s, va="top")
    text(ax, x, 0.905, TITLE["title"], F["sans_light"], TYPE["title"], INK, s, va="top")
    ax.plot([x, x + 0.085], [0.828, 0.828], transform=ax.transAxes, color=ACCENT["collider"],
            lw=1.6 * s, solid_capstyle="butt", zorder=8)
    text(ax, x, 0.805, TITLE["deck"], F["sans"], TYPE["deck"], INK_DIM, s, va="top")


def draw_meta(ax, F, s, anno):
    """Mono block: the facts that make the image checkable rather than decorative."""
    sky = anno.get("sky") or {}
    cam = anno.get("camera", {})
    loc = cam.get("location", [0, 0, 0])
    rows = [
        ("SITE", "Fermilab, Batavia IL  41.8319°N 88.2560°W"),
        ("INSTANT", f"{sky.get('utc', 'n/a')} UTC  —  {sky.get('twilight_phase', '')}"),
        ("SUN", f"{sky.get('sun_altitude_deg', float('nan')):+.1f}° alt  {sky.get('sun_azimuth_deg', float('nan')):.0f}° az"),
        ("GAL. CENTRE", f"{sky.get('galactic_centre_altitude_deg', float('nan')):.0f}° alt  {sky.get('galactic_centre_azimuth_deg', float('nan')):.0f}° az"),
        ("CAMERA", f"{cam.get('lens_mm', 0):.0f} mm  —  {loc[2]:.0f} m AGL"),
    ]
    scrim(ax, 0.0, 0.0, 0.46, 0.30, strength=0.60, direction="left")
    y = 0.175
    dy = 0.032
    for k, v in rows:
        text(ax, 0.038, y, k, F["mono_med"], TYPE["meta"], ACCENT["collider"], s, va="center", alpha=0.85)
        text(ax, 0.155, y, v, F["mono"], TYPE["meta"], INK_DIM, s, va="center")
        y -= dy
    text(ax, 0.038, y - 0.012, "Star field: NASA/GSFC SVS Deep Star Maps 2020 · Gaia DR2 (ESA/Gaia/DPAC)",
         F["mono"], TYPE["meta"] * 0.92, INK_DIM, s, va="center", alpha=0.75)


def draw_scale_and_north(ax, F, s, anno, width, height):
    """Scale bar and north arrow, both taken from the camera's own projection.

    A perspective view has no single scale, so the bar is measured at the site
    centroid and says so; that is honest where an unqualified bar would not be.
    """
    sc = anno.get("scale", {})
    px_per_km = sc.get("px_per_km_at_reference", 0.0)
    ang = np.radians(sc.get("screen_angle_deg_east", 0.0))
    if px_per_km <= 0:
        return
    scrim(ax, 0.58, 0.0, 1.0, 0.22, strength=0.55, direction="right")

    # --- scale bar: 2 km, drawn along the projected east direction ---------------
    km = 2.0
    length = km * px_per_km / width          # in axes-fraction of width
    x0, y0 = 0.655, 0.105
    dx, dy = length * np.cos(ang), length * np.sin(ang) * (width / height)
    ax.plot([x0, x0 + dx], [y0, y0 + dy], transform=ax.transAxes, color=INK, lw=1.8 * s,
            solid_capstyle="butt", zorder=8)
    for f in (0.0, 0.5, 1.0):
        tx, ty = x0 + dx * f, y0 + dy * f
        nx, ny = -np.sin(ang) * 0.009, np.cos(ang) * 0.009 * (width / height)
        ax.plot([tx - nx, tx + nx], [ty - ny, ty + ny], transform=ax.transAxes,
                color=INK, lw=1.4 * s, zorder=8)
    text(ax, x0, y0 - 0.045, f"0        1        {km:.0f} km", F["mono"], TYPE["scale"], INK, s, va="center")
    text(ax, x0, y0 - 0.082, "at site centre · perspective view", F["sans"], TYPE["metric"] * 0.95,
         INK_DIM, s, va="center", alpha=0.8)

    # --- north arrow ---------------------------------------------------------------
    na = np.radians(anno.get("north", {}).get("screen_angle_deg", 90.0))
    cx, cy, r = 0.945, 0.125, 0.030
    ux, uy = np.cos(na) * r, np.sin(na) * r * (width / height)
    ax.plot([cx - ux * 0.55, cx + ux], [cy - uy * 0.55, cy + uy], transform=ax.transAxes,
            color=INK, lw=1.5 * s, solid_capstyle="round", zorder=8)
    for sgn in (+1, -1):
        hx = cx + ux - (ux * 0.42) + sgn * (-uy * 0.30)
        hy = cy + uy - (uy * 0.42) + sgn * (ux * 0.30) * (height / width) * (width / height)
        ax.plot([cx + ux, hx], [cy + uy, hy], transform=ax.transAxes, color=INK,
                lw=1.5 * s, solid_capstyle="round", zorder=8)
    text(ax, cx + ux * 1.5, cy + uy * 1.5, "N", F["sans_semi"], TYPE["scale"], INK, s,
         ha="center", va="center")


def draw_features(ax, F, s, anno, layout_name, width, height):
    layout = LAYOUTS.get(layout_name, {})
    feats = anno.get("features", {})
    # far to near, so a near label overlaps a far one rather than the reverse
    order = sorted(feats.items(), key=lambda kv: -kv[1].get("depth_m", 0.0))
    for key, f in order:
        if key not in layout or not f.get("on_screen", False):
            continue
        L = layout[key]
        ax_x, ax_y = f["x"], 1.0 - f["y"]          # axes fraction has y up
        lx = ax_x + L["dx"]
        ly = ax_y + L["dy"]
        lx = float(np.clip(lx, 0.035, 0.965))
        ly = float(np.clip(ly, 0.055, 0.945))
        colour = ACCENT.get(f.get("accent", "neutral"), ACCENT["neutral"])
        ha = L.get("ha", "left")
        leader(ax, ax_x, ax_y, lx, ly, colour, s, ha=ha)
        pad = 0.010
        tx = lx + (pad if ha == "left" else -pad)
        text(ax, tx, ly + 0.016, f["label"], F["sans_med"], TYPE["label"], INK, s, ha=ha, va="center")
        if f.get("metric"):
            text(ax, tx, ly - 0.016, f["metric"], F["mono"], TYPE["metric"], colour, s,
                 ha=ha, va="center", alpha=0.92)


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
        draw_title(ax, F, s, anno.get("sky"))
        draw_meta(ax, F, s, anno)
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
