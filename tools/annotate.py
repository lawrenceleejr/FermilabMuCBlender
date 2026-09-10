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
LEADER = "#8D96A3"        # leaders are apparatus, not data: one neutral grey
ACCENT = {
    "collider": "#7FD2FF",     # proposed: collider, RCS, cooling, proton driver
    "tevatron": "#F2B173",     # existing plant, reused
    "boundary": "#F5F2EC",     # the site itself
    "offsite": "#B9B2D8",      # proposed but larger than the campus
}
# "the site" and "a building on the site" were two near-identical whites, so
# they were not two categories at all; anything uncategorised now reads as the
# site's own colour and the palette carries exactly the three it can support.
ACCENT["neutral"] = ACCENT["boundary"]

# Off-site is the figure's one real caveat, so it must survive a projector, a
# greyscale reprint and a colour-blind reader: it gets a dash pattern and a
# hollow marker, not just a hue.
DASHED = {"offsite"}

# The key, in the order a reader meets the argument. White is listed as an
# explicit category: with the site drawn in the same off-white as the body ink,
# an unlabelled white reads as a missing colour rather than as a class.
# Three entries, because there are now three categories. "Beyond the campus"
# was dropped along with the 35 km ring it described: a key entry with nothing
# in the frame to point at is worse than no key at all.
KEY = [
    ("collider", "Proposed"),
    ("tevatron", "Existing, reused"),
    ("boundary", "Site and context"),
]

# --- type scale -------------------------------------------------------------- #
# A genuine 1.25 scale, audited: 10.5, 13.1(unused), 15, 18.75, 23.4, 29.3, 36.6.
# The previous set claimed 1.25 in its comment and delivered ratios of 1.04-1.06
# between five sizes crammed into 8-12 pt, then a single 2.17x leap to the
# title -- so the figure had two hierarchy levels, not five, and everything
# below the title sat under the stroke floor for projection. Cap heights here
# are H/30 for the title and H/71 for a label, against H/42 and H/96 before.
SCALE_BASE = 1600.0
TYPE = {
    "title": 36.5,
    "deck": 18.5,
    "label": 15.0,
    "metric": 12.0,
    "meta": 12.0,       # legend numerals: the datum, not the caption
    "scale": 12.0,
    "credit": 10.5,     # a licence-required attribution has a legibility floor
}

# Metrics are quantities; the category belongs to the name that identifies the
# thing, not to its measurement. Colouring the metric put the categorical
# signal on the smallest, thinnest text in the figure and made the reader carry
# a hue back up a line, so every metric is now one dim neutral.
METRIC_INK = "#C9D2DE"

# --- per-camera label placement ---------------------------------------------- #
# A callout is either one feature key or a group: one leader to the first key's
# anchor, plain survey marks on the rest, one label for the set. Grouping is
# how nine rungs now cover eleven features -- three synchrotrons that share a
# story do not need three rungs of ladder.
#
# Sides and vertical order are *computed*, not typed. Every anchor in this view
# sits between x 0.20 and 0.60, so a fixed two-column assignment sent leaders
# back across the frame: nine pairs crossed. Assigning each callout to the
# column its own anchor is nearer, then ordering each column by anchor height,
# makes crossings geometrically impossible -- within a column the order is
# monotonic, and between columns the leaders travel in opposite directions.
#
# That costs the beam order, which stacking used to imply badly. It is now
# stated outright in the sequence ribbon along the bottom, so the reader gets
# the chain from a line that actually says 1 -> 6 rather than from guessing at
# the reading order of a ladder.
CHAIN = [
    dict(keys=["proton_driver"], stage=1),
    dict(keys=["cooling"], stage=2),
    dict(keys=["rcs12", "rcs3"], stage=3, label="RCS 1\u20133",
         metric="rapid-cycling, 5.99 and 10.7\u202fkm"),
    dict(keys=["rcs4"], stage=4),
    dict(keys=["collider"], stage=5),
    dict(keys=["detector_a", "detector_b"], stage=6, label="Detector halls",
         metric="two interaction points"),
]
CONTEXT = [
    dict(keys=["boundary"]),
    dict(keys=["tevatron"]),
    dict(keys=["main_injector"]),
]
# The chain, in beam order, for the ribbon. Short forms: the ribbon is a
# sequence, not a set of definitions.
RIBBON = [(1, "Proton driver"), (2, "Cooling"), (3, "RCS 1\u20133"),
          (4, "RCS 4 filler"), (5, "Collider"), (6, "Detectors")]

LAYOUTS = {
    "overview": CHAIN + CONTEXT,
    "northeast": [dict(keys=["collider"], stage=5), dict(keys=["wilson"])],
}

COLUMN_SPLIT = 0.47          # anchors left of this get the left column


def band(anchors, n, *, min_gap=0.058, floor=None, ceil=None):
    """Vertical extent for a ladder of n rungs, scaled to its own anchor field.

    The previous version spread the labels over a fixed 51 % of frame height
    while every anchor sat inside a 29 % band -- a 1.76x stretch, which is what
    forced the legs to fan. Sizing the ladder from the anchors it serves keeps
    the legs short and nearly parallel.
    """
    ceil = TITLE_FLOOR - 0.030 if ceil is None else ceil
    floor = LABEL_FLOOR if floor is None else floor
    need = max((n - 1) * min_gap, 1e-6)
    if anchors:
        lo, hi = min(anchors), max(anchors)
    else:
        lo = hi = (floor + ceil) / 2
    mid = (lo + hi) / 2
    half = max((hi - lo) / 2, need / 2)
    top, bot = mid + half, mid - half
    # keep it inside the reserved region, sliding rather than stretching
    if top > ceil:
        bot -= top - ceil
        top = ceil
    if bot < floor:
        top = min(ceil, top + (floor - bot))
        bot = floor
    return bot, top


def ladder(anchors, n):
    """n evenly spaced label baselines, top first, over the fitted band."""
    lo, hi = band(anchors, n)
    if n <= 1:
        return [(lo + hi) / 2]
    step = (hi - lo) / (n - 1)
    return [hi - i * step for i in range(n)]


TITLE = dict(
    title="A Future Muon Collider at Fermilab",
    # The standfirst has to state what the geometry supports, and the geometry
    # changed: sizing the final synchrotron to the site rather than to the
    # IMCC's 35 km reference puts the whole chain inside the boundary.
    deck="The whole accelerator chain, sized to the existing 27 km\u00b2 campus \u2014 "
         "its final synchrotron a 14.5 km ring that fits.",
)

# The footer carries only what the image cannot say for itself: where the
# numbers came from, what the drawing is not, and the credits the data licences
# ask for. The instant, solar geometry and lens that used to sit here were
# detail a site plan's reader does not need.
PROVENANCE = ("Machine parameters: IMCC Tentative Parameter List, 30 Oct 2023 (10 TeV option); "
              "the final synchrotron is sized to the site at 14.5 km rather than the 35 km reference, "
              "which needs more acceleration turns. Siting indicative, not an engineering study.")
CREDIT = ("Procedural Cycles render \u00b7 terrain and site data \u00a9 OpenStreetMap contributors (ODbL) "
          "and AWS Terrain Tiles \u00b7 star field NASA/GSFC SVS Deep Star Maps 2020, Gaia DR2 (ESA/Gaia/DPAC)")

# One margin and one set of horizontal bands, so every block aligns to the same
# edges and the label spacing is computed rather than typed.
MARGIN = 0.042
GUTTER = 0.030               # the common knee gutter: one vertical spine per column

# The bottom of the frame is stacked explicitly from the bottom edge upward,
# because everything in it has variable height: the footer wraps to as many
# lines as the copy needs, and the graphic scale's legs are however long the
# projection makes them. Deriving each band from the one below it is what stops
# the footer growing into the legend, which is what happened when the rule was
# a fixed 0.052 and the provenance line grew.
FOOTER_BOTTOM = 0.016        # baseline of the lowest footer line
FOOTER_LEAD = 0.020          # leading within the footer block
FOOTER_LINES = 4             # reserved; the block wraps into at most this many
FOOTER_RULE = FOOTER_BOTTOM + FOOTER_LINES * FOOTER_LEAD + 0.014
LEGEND_BAND = (FOOTER_RULE + 0.014, FOOTER_RULE + 0.130)
TITLE_FLOOR = 0.800          # nothing else goes above this on the left
# The columns need to be wide enough for the longest metric string; an earlier
# 0.098 gave 157 px for a 187 px string, so six blocks hung past the margin.
COL_X = (MARGIN + 0.174, 1.0 - MARGIN - 0.174)
LABEL_FLOOR = LEGEND_BAND[1] + 0.045   # the ladder may not reach into the legend


# --------------------------------------------------------------------------- #
# fonts
# --------------------------------------------------------------------------- #
def load_fonts() -> dict[str, fm.FontProperties]:
    """Register the self-hosted IBM Plex TTFs and return FontProperties by role.

    One family throughout -- IBM Plex Sans, which was designed for technical
    contexts and holds up at label sizes. Hierarchy comes from weight and size
    alone, which is steadier than mixing in a second face.

    The 300 weight is deliberately not loaded. It was here for exactly one
    string, the title, and at 36 px it delivered 2 px stems inside a 1.5 px
    halo -- the lightest, most fragile element in the frame was also the
    largest, which inverts the hierarchy it was meant to establish. Display
    separation now comes from size, at a weight the medium can hold.
    """
    faces = {
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
def scrim(ax, x0, y0, x1, y1, *, strength=0.62, direction="left", feather=0.40, zorder=2):
    """A soft directional gradient behind a text block.

    Type over a photographic background needs separation. A gradient scrim keeps
    the image visible and reads as intentional, where a hard box or a per-letter
    outline reads as a screenshot.

    The whole justification for a scrim is that it has no locatable edge, and
    the previous version had one: the ramp fell to zero on the far side only,
    so its other three sides ended in a hard step -- measured at +19 luminance
    across 66 % of the frame width, plainly visible as a pasted rectangle. Every
    edge that is not a frame edge is now feathered, on both axes, so alpha
    reaches zero before the scrim ends. `feather` is the fraction of the extent
    given over to each fade.
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

    def profile(lo_at_frame, hi_at_frame):
        """1 in the middle, fading to 0 at whichever ends sit inside the frame."""
        p = np.ones(n)
        if not lo_at_frame:
            p *= np.clip(ramp / feather, 0.0, 1.0)
        if not hi_at_frame:
            p *= np.clip((1.0 - ramp) / feather, 0.0, 1.0)
        return p

    eps = 1e-6
    if feather > 0:
        a = a * profile(y0 <= eps, y1 >= 1.0 - eps)[:, None]
        a = a * profile(x0 <= eps, x1 >= 1.0 - eps)[None, :]

    rgba = np.zeros((n, n, 4))
    rgba[..., 3] = a * strength
    ax.imshow(rgba, extent=(x0, x1, y0, y1), transform=ax.transAxes,
              origin="lower", aspect="auto", zorder=zorder, interpolation="bilinear")


LEADERS: list[tuple[str, tuple, tuple, tuple]] = []   # (key, anchor, knee, end) in axes coords


def leader(ax, ax_x, ax_y, lx, ly, colour, s, width, height, *, ha="left", key="",
           dashed=False, zorder=6):
    """Dog-leg leader: a diagonal off the anchor into a common knee gutter,
    then a short horizontal run that underlines the text block.

    Two things were wrong with the previous version. It drew at the same weight
    and in the same accent hue as the accelerator geometry, so where a cyan
    leader crossed a cyan tunnel the reader could not tell the pointer from the
    subject -- a leader is apparatus and must be subordinate to the thing it
    points at. And it kicked at a fixed 45 degrees, which put every knee at a
    different x and gave horizontal runs from 70 to 615 px, four of them
    crossing the whole eastern half of the complex. Launching every diagonal
    from one vertical spine, `GUTTER` outboard of the text, makes each run the
    same short length and drops the crossings.

    The run terminates at the metric's baseline rather than between the label
    and the metric, so it binds the pair from below instead of splitting it.
    """
    towards = 1.0 if ha == "left" else -1.0     # which way the label lies
    knee_x = lx - towards * GUTTER
    # if the anchor is already past the gutter, there is no diagonal to draw
    if (towards > 0 and ax_x > knee_x) or (towards < 0 and ax_x < knee_x):
        knee_x = ax_x
    ax.plot([ax_x, knee_x, lx], [ax_y, ly, ly], transform=ax.transAxes, color=LEADER,
            alpha=0.55, lw=0.6 * s, solid_capstyle="round", solid_joinstyle="miter",
            dashes=(4, 3) if dashed else (None, None), zorder=zorder,
            path_effects=[patheffects.withStroke(linewidth=1.8 * s, foreground="#05070B66")])
    LEADERS.append((key, (ax_x, ax_y), (knee_x, ly), (lx, ly)))
    # the survey mark keeps the category colour: it is the one place the leader
    # touches its subject, so it is where the classification belongs
    if dashed:
        ax.plot([ax_x], [ax_y], transform=ax.transAxes, marker="o", ms=4.4 * s,
                mfc="none", mec=colour, mew=1.1 * s, zorder=zorder + 1)
    else:
        ax.plot([ax_x], [ax_y], transform=ax.transAxes, marker="o", ms=3.2 * s,
                mfc=colour, mec="none", zorder=zorder + 1)
    ax.plot([ax_x], [ax_y], transform=ax.transAxes, marker="o", ms=9.0 * s,
            mfc="none", mec=colour, mew=0.7 * s, alpha=0.40, zorder=zorder)


def dot(ax, ax_x, ax_y, colour, s, *, dashed=False, zorder=7):
    """A bare survey mark: a grouped callout's second and later anchors, which
    share one label and so must not each drag their own leader across the frame."""
    ax.plot([ax_x], [ax_y], transform=ax.transAxes, marker="o", ms=3.2 * s,
            mfc="none" if dashed else colour, mec=colour, mew=1.1 * s, zorder=zorder)


DRAWN: list[tuple[str, str, object]] = []   # (role, group, artist) for the checker
RENDERER = [None]     # set in build(); lets a run of type be measured as it is laid out


def fit_lines(ax, body, fp, size, s, max_w, width):
    """Break `body` into lines no wider than `max_w` axes fractions.

    Greedy, on real measured extents rather than a character count: the
    provenance line grew when it took on the site-filler explanation and ran
    221 px past the right margin, which a character budget would not have
    caught because it depends on the face and the size.
    """
    words = body.split()
    lines, cur = [], ""
    for wd in words:
        trial = f"{cur} {wd}".strip()
        probe = ax.text(0, -1, trial, transform=ax.transAxes, fontproperties=fp,
                        fontsize=size * s)
        w_ax = measure(probe, width)
        probe.remove()
        if cur and w_ax > max_w:
            lines.append(cur)
            cur = wd
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def measure(t, width) -> float:
    """Axes-fraction width of an already-created text artist.

    Laying out a horizontal run needs the real advance of each item. Estimating
    it as a constant per character is what made the sequence ribbon's arrows
    sit at visibly unequal gaps: "Cooling" and "RCS 1-3" are the same length in
    characters and not in ink.
    """
    r = RENDERER[0]
    if r is None:
        return 0.0
    return t.get_window_extent(r).width / width


def text(ax, x, y, body, fp, size, colour, s, *, ha="left", va="center", zorder=8,
         shadow=True, alpha=1.0, role="text", group=None):
    """One text block, with a halo proportional to the type it protects.

    The halo used to be a fixed 2.2 px at every size, which is 8.5 % of the em
    on the title and 23 % on a metric -- so the small type sat in a black furry
    box with its counters clogged, and the accent colour never reached its
    specified value. Scaling it with the size makes it one policy rather than
    three accidental ones.
    """
    t = ax.text(x, y, body, transform=ax.transAxes, ha=ha, va=va,
                fontproperties=fp, fontsize=size * s, color=colour, zorder=zorder, alpha=alpha)
    # `group` marks blocks that are meant to sit close -- the lines of one
    # paragraph, a name and its metric -- so the checker does not read
    # deliberate leading as a collision. Everything else needs full clearance.
    DRAWN.append((role, group or role, t))
    if shadow:
        lw = min(0.13 * size * s, 2.6 * s)
        t.set_path_effects([patheffects.withStroke(linewidth=lw, foreground="#05070B99")])
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
    and dropping it lets the title hold the top of the frame on its own.

    The block sits high enough to clear the horizon glow. The deck used to be
    set at 12 pt in a cool grey directly on the brightest band of the render,
    which made the line carrying the figure's actual claim the least legible
    text on the page; it is now the second-largest thing in the frame, on a
    feathered scrim, with no per-letter stroke fighting it.
    """
    copy = copy or TITLE
    scrim(ax, 0.0, 0.52, 0.70, 1.0, strength=0.74, direction="left")
    x = MARGIN
    text(ax, x, 0.950, copy["title"], F["sans"], TYPE["title"], INK, s, va="top",
         shadow=False, role="title")
    # a measured tick, not an orphaned underline: the old 99 px rule under a
    # 560 px title read as neither. Neutral, so the title block does not enrol
    # itself in the categorical colour scale.
    ax.plot([x, x + 0.150], [0.874, 0.874], transform=ax.transAxes, color=INK,
            lw=1.6 * s, alpha=0.35, solid_capstyle="butt", zorder=8)
    text(ax, x, 0.852, copy["deck"], F["sans"], TYPE["deck"], "#D8D3C9", s, va="top",
         shadow=False, role="deck")


def draw_ribbon(ax, F, s, width):
    """The beam sequence, stated rather than implied.

    The callout ladder is ordered by geometry so its leaders cannot cross,
    which means the stacking order carries no sequence. This line carries it:
    a figure whose subject is a chain has to say, somewhere, which end the beam
    enters. One line of type does what twelve leaders could not.
    """
    y = LEGEND_BAND[1] - 0.010
    x = MARGIN
    gap, tight = 0.020, 0.006
    t = text(ax, x, y, "Beam sequence", F["sans_semi"], TYPE["credit"], INK_DIM, s,
             va="center", shadow=False, role="legend:ribbon", group="ribbon")
    x += measure(t, width) + gap
    for i, (n, name) in enumerate(RIBBON):
        if i:
            t = text(ax, x, y, "\u2192", F["sans"], TYPE["credit"], INK_DIM, s,
                     va="center", shadow=False, role="legend:ribbon", group="ribbon")
            x += measure(t, width) + gap
        t = text(ax, x, y, f"{n}", F["sans_semi"], TYPE["credit"], ACCENT["collider"], s,
                 va="center", shadow=False, role="legend:ribbon", group="ribbon")
        x += measure(t, width) + tight
        t = text(ax, x, y, name, F["sans"], TYPE["credit"], INK, s,
                 va="center", shadow=False, role="legend:ribbon", group="ribbon")
        x += measure(t, width) + gap
    return x


def draw_key(ax, F, s, width):
    """The categorical key, on one line at the left of the legend band.

    A colour category with no key is decoration. Off-site carries a dash
    pattern as well as a hue, so the one real caveat in the figure survives a
    projector, a greyscale reprint and a colour-blind reader.
    """
    y = LEGEND_BAND[0] + 0.014
    x = MARGIN
    swatch, gap = 0.020, 0.030
    for accent, caption in KEY:
        colour = ACCENT[accent]
        ax.plot([x, x + swatch], [y, y], transform=ax.transAxes, color=colour,
                lw=2.2 * s, solid_capstyle="butt", zorder=8,
                dashes=(3, 2) if accent in DASHED else (None, None))
        t = text(ax, x + swatch + 0.008, y, caption, F["sans"], TYPE["credit"], INK, s,
                 va="center", shadow=False, role="legend:key", group="key")
        x += swatch + 0.008 + measure(t, width) + gap
    return x


def draw_footer(ax, F, s, anno, width):
    """One hairline, then provenance and credits, wrapped to the margins.

    Where the numbers came from is part of the data: without it the figure is
    an assertion. The credit line used to be dimmed twice over -- a grey ink
    *and* 55 % alpha -- compositing to a contrast ratio of 3.0 at a 7.8 px cap
    height, which fails the only function an attribution has.

    Both blocks are wrapped on measured widths. The footer is the one place
    where copy grows as the work is explained, so a fixed single line here is a
    margin overrun waiting to happen -- and it happened.
    """
    ax.plot([MARGIN, 1.0 - MARGIN], [FOOTER_RULE, FOOTER_RULE], transform=ax.transAxes,
            color=INK, lw=1.0 * s, alpha=0.28, zorder=7)
    avail = 1.0 - 2 * MARGIN
    lines = []
    for body, colour, role in ((PROVENANCE, "#A7AEB9", "credit:provenance"),
                               (CREDIT, "#8C939E", "credit:sources")):
        for line in fit_lines(ax, body, F["sans"], TYPE["credit"], s, avail, width):
            lines.append((line, colour, role))
    if len(lines) > FOOTER_LINES:
        print(f"[annotate] footer wrapped to {len(lines)} lines but only "
              f"{FOOTER_LINES} are reserved; raise FOOTER_LINES or shorten the copy")
    # laid out upward from the bottom margin, so the block cannot run off the
    # frame however long the copy gets
    y = FOOTER_BOTTOM + (len(lines) - 1) * FOOTER_LEAD
    for line, colour, role in lines:
        text(ax, MARGIN, y, line, F["sans"], TYPE["credit"], colour, s,
             va="center", shadow=False, role=role, group="footer")
        y -= FOOTER_LEAD


def draw_scale_and_north(ax, F, s, anno, width, height):
    """A two-axis graphic scale, right-aligned with the margin.

    A perspective view has no single scale, and this one is anisotropic: at the
    site centre 1 km east projects to 134.6 px while 1 km north projects to
    56.9 px. A single horizontal bar calibrated on the east rate -- which is
    what was here -- makes the reader measure the collider ring's north-south
    extent as 1.34 km when it is 3.18 km. That is not a caveat, it is a wrong
    number, and the "at site centre" caption addressed the smaller error while
    concealing the larger one.

    So the bar becomes an L: 2 km east along the bottom and 2 km north up the
    side, both at their true projected lengths, sharing a zero corner. The
    foreshortening becomes visible instead of hidden, and the north leg carries
    the bearing, which makes a separate north needle redundant.
    """
    sc = anno.get("scale", {})
    px_per_km = sc.get("px_per_km_at_reference", 0.0)
    if px_per_km <= 0:
        return
    nx, ny = axes_dir(anno.get("north", {}).get("north_px_per_km", [0.0, -px_per_km]), width, height)
    # Pick the round distance that fits the band rather than assuming one. This
    # camera projects 1 km to 206 px east and 95 px north, so the 2 km the
    # legend used to assume would be a 412 px bar with a 191 px riser -- taller
    # than the legend band and straight through the callout ladder.
    km = 1.0
    for cand in (5.0, 2.0, 1.0, 0.5):
        if (cand * px_per_km / width <= 0.20
                and abs(cand * ny) <= (LEGEND_BAND[1] - LEGEND_BAND[0]) * 0.80):
            km = cand
            break
    east = km * px_per_km / width
    ndx, ndy = km * nx, km * ny

    # one shared right edge with the margin and the footer rule: three
    # near-identical right edges 20-140 px apart read as sloppiness
    x1 = 1.0 - MARGIN
    x0 = x1 - east
    y0 = LEGEND_BAND[0] + 0.010

    ax.plot([x0, x1], [y0, y0], transform=ax.transAxes, color=INK, lw=1.9 * s,
            solid_capstyle="butt", zorder=8)
    ax.plot([x0, x0 + ndx], [y0, y0 + ndy], transform=ax.transAxes, color=INK, lw=1.9 * s,
            solid_capstyle="butt", zorder=8)
    for f in (0.5, 1.0):                                # ticks on the east leg
        tx = x0 + east * f
        ax.plot([tx, tx], [y0, y0 - 0.012], transform=ax.transAxes, color=INK, lw=1.4 * s, zorder=8)
    text(ax, x1, y0 - 0.034, f"{km:g}\u202fkm east", F["sans_med"], TYPE["meta"], INK, s,
         ha="right", va="center", shadow=False, role="legend:scale", group="scale")
    text(ax, x0 + ndx + 0.007, y0 + ndy, f"{km:g}\u202fkm north", F["sans_med"], TYPE["meta"],
         INK, s, ha="left", va="center", shadow=False, role="legend:scale", group="scale")
    text(ax, x1, y0 - 0.054, "projected at site centre",
         F["sans"], TYPE["credit"], INK_DIM, s, ha="right", va="center", shadow=False, role="legend:scale", group="scale")


def draw_features(ax, F, s, anno, layout_name, width, height):
    """Place the callout ladders, choosing sides and order from the geometry."""
    groups = []
    feats = anno.get("features", {})
    for g in LAYOUTS.get(layout_name, []):
        keys = [k for k in g["keys"] if k in feats and feats[k].get("on_screen")]
        if keys:
            groups.append((g, keys))

    # Split by rank, not by a fixed x. Nine of the twelve anchors in this view
    # sit left of centre, so a fixed COLUMN_SPLIT put eight callouts in the left
    # column and one in the right: the left ladder then had to stretch far from
    # its own anchors, which is what reintroduced a crossing. Ranking by anchor
    # x and halving keeps the columns within one of each other while still
    # giving every callout the side its anchor leans toward.
    ordered = sorted(groups, key=lambda t: feats[t[1][0]]["x"])
    half = (len(ordered) + 1) // 2
    columns = {"left": ordered[:half], "right": ordered[half:]}

    placed = []
    for side, items in columns.items():
        # tallest anchor to the top rung: monotonic within a column, so the
        # diagonals cannot cross each other
        items.sort(key=lambda t: -(1.0 - feats[t[1][0]]["y"]))
        anchors = [1.0 - feats[keys[0]]["y"] for _, keys in items]
        ys = ladder(anchors, len(items))
        lx = COL_X[0] if side == "left" else COL_X[1]
        ha = "right" if side == "left" else "left"
        for (g, keys), ly in zip(items, ys):
            placed.append((g, keys, lx, ly, ha, feats[keys[0]].get("depth_m", 0.0)))

    # far to near, so a nearer callout draws over a farther one
    for g, keys, lx, ly, ha, _ in sorted(placed, key=lambda t: -t[5]):
        prim = feats[keys[0]]
        accent = g.get("accent", prim.get("accent", "neutral"))
        colour = ACCENT.get(accent, ACCENT["neutral"])
        dashed = accent in DASHED
        ax_x, ax_y = prim["x"], 1.0 - prim["y"]
        pad = 0.010
        tx = lx + (pad if ha == "left" else -pad)
        # the pair is tightened so it reads as one unit against the ladder step
        y_name, y_metric = ly + 0.012, ly - 0.012
        leader(ax, ax_x, ax_y, lx, y_metric, colour, s, width, height, ha=ha,
               key=keys[0], dashed=dashed)
        prev = (ax_x, ax_y)
        for k in keys[1:]:
            px, py = feats[k]["x"], 1.0 - feats[k]["y"]
            # a hairline tie, so the set reads as one callout rather than as a
            # labelled anchor plus an unexplained dot elsewhere in the frame
            ax.plot([prev[0], px], [prev[1], py], transform=ax.transAxes, color=colour,
                    lw=0.5 * s, alpha=0.35, dashes=(2, 3), zorder=5)
            dot(ax, px, py, colour, s, dashed=dashed)
            prev = (px, py)

        label = g.get("label") or prim["label"]
        if g.get("stage"):
            label = f"{g['stage']}\u2003{label}"
        # the category rides the name: it is the identifier, and the more
        # prominent half of the pair
        text(ax, tx, y_name, label, F["sans_semi"], TYPE["label"], colour, s, ha=ha,
             va="center", role=f"label:{keys[0]}", group=f"callout:{keys[0]}")
        metric = g.get("metric") or prim.get("metric")
        if metric:
            text(ax, tx, y_metric, metric, F["sans"], TYPE["metric"], METRIC_INK, s,
                 ha=ha, va="center", role=f"metric:{keys[0]}", group=f"callout:{keys[0]}")

    wanted = {k for g in LAYOUTS.get(layout_name, []) for k in g["keys"]}
    missing = [k for k in wanted if k not in feats or not feats[k].get("on_screen")]
    if missing:
        print(f"[annotate] not drawn (off screen or absent): {', '.join(sorted(missing))}")
    return len(placed)


def _crosses(p1, p2, p3, p4):
    """True if segment p1-p2 properly crosses p3-p4 (pixel coordinates)."""
    def o(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return 0 if abs(v) < 1e-9 else (1 if v > 0 else -1)
    d1, d2 = o(p3, p4, p1), o(p3, p4, p2)
    d3, d4 = o(p1, p2, p3), o(p1, p2, p4)
    return d1 != d2 and d3 != d4 and 0 not in (d1, d2, d3, d4)


def check_collisions(fig, ax, width, height, *, pad_px=6.0):
    """Measure the layout and report everything that is actually wrong with it.

    The previous version tested pairwise text overlap and a 2 px canvas clip,
    and printed "spacing OK" for a figure in which six blocks hung up to 48 px
    past the left margin and seven pairs of leaders crossed. Passing its own QA
    while breaking its own margin is the failure mode this function exists to
    prevent, so it now checks the things that were wrong:

      * text against text, and against the reserved bands
      * every block against the margin, not just the trim
      * leader against leader, as real segment intersections
      * anything running off the canvas

    A guard that measures the wrong quantity is worse than no guard, because it
    licenses the belief that the layout has been checked.
    """
    # a bare Figure has no renderer until one is attached; Agg gives real
    # FreeType extents, which is the whole point of measuring rather than guessing
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    boxes = []
    for role, grp, t in DRAWN:
        bb = t.get_window_extent(renderer)
        boxes.append((role, bb.x0, bb.y0, bb.x1, bb.y1, grp))

    def overlap(a, b, pad):
        return not (a[3] + pad <= b[1] or b[3] + pad <= a[1]
                    or a[4] + pad <= b[2] or b[4] + pad <= a[2])

    hits = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            pad = 1.0 if a[5] == b[5] else pad_px       # same group: close is intended
            if overlap(a, b, pad):
                hits.append((a[0], b[0], "text/text"))

    # text against the reserved bands, in pixels
    zones = {
        "footer band": (0.0, 0.0, 1.0, FOOTER_RULE),
        "legend band": (0.0, LEGEND_BAND[0] - 0.012, 1.0, LEGEND_BAND[1]),
    }
    for role, x0, y0, x1, y1, _grp in boxes:
        if role.startswith(("credit", "legend")):
            continue
        for zname, (zx0, zy0, zx1, zy1) in zones.items():
            z = (zname, zx0 * width, zy0 * height, zx1 * width, zy1 * height)
            if overlap((role, x0, y0, x1, y1), z, 2.0):
                hits.append((role, zname, "text/zone"))

    # the margin is a specification, not an outcome: a flush edge that some
    # lines cross is not an edge
    ml, mr = MARGIN * width, (1.0 - MARGIN) * width
    for role, x0, y0, x1, y1, _grp in boxes:
        if x0 < ml - 1.0:
            hits.append((role, f"left margin by {ml - x0:.0f} px", "margin"))
        if x1 > mr + 1.0:
            hits.append((role, f"right margin by {x1 - mr:.0f} px", "margin"))

    # leader against leader: a crossing is where the reader loses the thread,
    # and it is the one thing the old checker could not have caught
    segs = []
    for key, a, k, e in LEADERS:
        pa = (a[0] * width, a[1] * height)
        pk = (k[0] * width, k[1] * height)
        pe = (e[0] * width, e[1] * height)
        segs.append((key, [(pa, pk), (pk, pe)]))
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            k1, s1 = segs[i]
            k2, s2 = segs[j]
            for a1, b1 in s1:
                for a2, b2 in s2:
                    if _crosses(a1, b1, a2, b2):
                        hits.append((f"leader:{k1}", f"leader:{k2}", "cross"))
                        break
                else:
                    continue
                break

    # anything running off the canvas
    for role, x0, y0, x1, y1, _grp in boxes:
        if x0 < 2 or y0 < 2 or x1 > width - 2 or y1 > height - 2:
            hits.append((role, "canvas edge", "clipped"))

    if hits:
        print(f"[annotate] {len(hits)} layout problem(s):")
        for a, b, kind in hits:
            print(f"    {kind:11s} {a}  <->  {b}")
    else:
        print(f"[annotate] layout OK: {len(boxes)} text blocks and {len(LEADERS)} leaders; "
              f"no overlaps within {pad_px:.0f} px, no margin breaks, no leader crossings")
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

    DRAWN.clear()
    LEADERS.clear()
    s = width / SCALE_BASE          # keeps type and rules proportional at any size
    dpi = 100.0
    F = load_fonts()

    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    # attaching a canvas up front gives FreeType extents during layout, not
    # just afterwards in the checker
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    RENDERER[0] = FigureCanvasAgg(fig).get_renderer()
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
            text(ax, x + 0.006, y, f"{key}  ({x:.3f},{f['y']:.3f})", F["sans"], 8.5, "#FF3B30", s,
                 role=f"debug:{key}")
        for gx in np.arange(0.1, 1.0, 0.1):
            ax.axvline(gx, color="#FF3B3055", lw=0.6 * s, zorder=4)
            ax.axhline(gx, color="#FF3B3055", lw=0.6 * s, zorder=4)
    else:
        draw_title(ax, F, s, copy=dict(
            title=args.title or TITLE["title"],
            deck=(args.deck.replace("\\n", "\n") if args.deck else TITLE["deck"]),
        ))
        draw_footer(ax, F, s, anno, width)
        draw_ribbon(ax, F, s, width)
        draw_key(ax, F, s, width)
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
