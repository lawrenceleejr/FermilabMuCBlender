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
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.font_manager as fm  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
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
# Raised from #9AA3AF, which measured a WCAG ratio of 3.95 against the ground
# actually behind it -- the three blocks set in it (the ribbon caption, the
# ribbon arrows, the scale-bar note) were the least legible type in the figure,
# and darkening the ground under them any further would have cost more picture
# than the reading was worth. At 0.50 relative luminance it stays a clear step
# below METRIC_INK, so the rank the dim grey was chosen to carry survives.
INK_DIM = "#B4BCC6"
# Leaders stay one neutral grey -- apparatus, not data, so they never compete
# with the accelerator geometry for hue. But at 0.6 px and 55 % alpha they had
# gone too far the other way and were barely findable, so the weight and
# opacity come back up: the elbow has to be traceable from label to subject at
# a glance, which is the whole point of a dog-leg.
LEADER = "#C3CBD6"
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
# column its own anchor is nearer, which keeps leaders off the drawing. Within
# a column, ordering by anchor height is NOT sufficient -- an earlier version of
# this comment claimed it made crossings impossible, and it does not: two
# leaders converging on one gutter cross as soon as their anchors differ in x.
# The order is settled by the crossing test itself, in draw_features.
#
# That costs the beam order, which stacking used to imply badly. It is now
# stated outright in the sequence ribbon along the bottom, so the reader gets
# the chain from a line that actually says 1 -> 6 rather than from guessing at
# the reading order of a ladder.
# Each callout carries its own name and its own purpose line. The purpose line
# replaced a metric ("6.28 km, reused"), and the trade is deliberate: a reader
# looking at a site plan can see that one ring is bigger than another, and
# cannot see what either is for. Where a number is the point it is folded into
# the sentence rather than given a line of its own.
#
# Names are title case, and the stage numbers are the only sequence cue left
# now that the beam-sequence ribbon is gone -- which is why the ladder is
# ordered by stage rather than by geometry.
CHAIN = [
    dict(keys=["proton_driver"], stage=1, label="Proton Driver",
         purpose="MW-level proton beam incident on target; "
                 "resulting pions decay to muons"),
    dict(keys=["cooling"], stage=2, label="Muon Cooling",
         purpose="Muon beam phase space volume reduction via ionization cooling"),
    # All four synchrotrons on one rung. They were two callouts, split on where
    # they sit -- 1-3 inside the Tevatron, 4 filling the site -- but they are one
    # stage of the beam and one sentence describes all of them, and nine callouts
    # of wrapped copy do not fit two columns without either eating the drawing
    # or dropping below the legend band. The survey marks still mark all four.
    dict(keys=["rcs4", "rcs12", "rcs3"], stage=3, label="RCS 1\u20134",
         purpose="Rapid cycling synchrotrons to accelerate beam to TeV scale"),
    dict(keys=["collider"], stage=4, label="Collider Ring",
         purpose="Storage rings lead to 10 TeV collisions in detector halls"),
    dict(keys=["detector_a", "detector_b"], stage=5, label="Detector Halls",
         purpose="Cathedral-sized detectors measure collision products "
                 "in presence of large beam background"),
]
CONTEXT = [
    dict(keys=["boundary"], label="Fermilab Site",
         purpose="27.7 km\u00b2 campus contains the entire chain"),
    dict(keys=["tevatron"], label="Tevatron Ring",
         purpose="Existing 6.28 km tunnel, reused for the first synchrotrons"),
    dict(keys=["main_injector"], label="Main Injector",
         purpose="Existing 3.32 km synchrotron, available for reuse"),
]

LAYOUTS = {
    "overview": CHAIN + CONTEXT,
    "northeast": [
        dict(keys=["collider"], stage=5, label="Collider Ring",
             purpose="Storage rings lead to 10 TeV collisions in detector halls"),
        dict(keys=["wilson"], label="Wilson Hall",
             purpose="Central laboratory and site landmark"),
    ],
}

COLUMN_SPLIT = 0.47          # anchors left of this get the left column


def band(anchors, n, *, min_gap=0.058, floor=None, ceil=None):   # noqa: retained for reference
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


def place_rungs(anchors, *, min_gap=0.058, lift=None, floor=None, ceil=None):
    """A rung per anchor, lifted above it, pushed apart only as much as the
    minimum gap demands.

    An evenly spaced ladder is tidy and wrong. It puts rungs far from their
    anchors, which makes leaders long, makes them traverse the text column, and
    -- when the ladder is taller than the anchor field -- makes them cross.
    Starting each label level with the thing it names and separating only where
    two would collide keeps every leader as short as it can be, which is also
    the better answer typographically: a label belongs next to its subject.

    But level with its anchor is one step too far: the diagonal leg disappears
    and the leader is a plain horizontal rule. `lift` raises the whole ladder by
    a constant so the dog-leg has an angle again, without putting the rungs
    back out of order.

    Returns the rungs in the same order as `anchors`.
    """
    ceil = TITLE_FLOOR - 0.030 if ceil is None else ceil
    floor = LABEL_FLOOR if floor is None else floor
    lift = LABEL_LIFT if lift is None else lift
    n = len(anchors)
    if n == 0:
        return []
    anchors = [a + lift for a in anchors]
    span = (n - 1) * min_gap
    if span > ceil - floor:                     # more rungs than room: spread evenly
        step = (ceil - floor) / max(n - 1, 1)
        order = sorted(range(n), key=lambda i: -anchors[i])
        out = [0.0] * n
        for r, i in enumerate(order):
            out[i] = ceil - r * step
        return out

    order = sorted(range(n), key=lambda i: -anchors[i])      # top first
    ys = [min(max(anchors[i], floor), ceil) for i in order]
    # separate downward, then recover off the floor, then off the ceiling
    for _ in range(4):
        for k in range(1, n):
            ys[k] = min(ys[k], ys[k - 1] - min_gap)
        if ys[-1] < floor:
            ys[-1] = floor
            for k in range(n - 2, -1, -1):
                ys[k] = max(ys[k], ys[k + 1] + min_gap)
        if ys[0] > ceil:
            shift = ys[0] - ceil
            ys = [y - shift for y in ys]
    out = [0.0] * n
    for k, i in enumerate(order):
        out[i] = ys[k]
    return out


TITLE = dict(
    title="A Future Muon Collider at Fermilab",
    # The standfirst has to state what the geometry supports, and the geometry
    # changed: sizing the final synchrotron to the site rather than to the
    # IMCC's 35 km reference puts the whole chain inside the boundary.
    # No area figure here: the boundary callout carries the measured 27.7 km2,
    # and a rounded 27 in the deck against a measured 28 in the margin is
    # exactly the kind of internal disagreement a reader can catch.
    # Broken at the comma, which also keeps both lines inside the 45-75
    # character measure that is comfortable to read: 57 and 72.
    deck="An accelerator and collider complex on the Fermilab site,\n"
         "reusing existing infrastructure, targeting collision energies of 10 TeV.",
)

# The footer carries only what the image cannot say for itself: where the
# numbers came from, what the drawing is not, and the credits the data licences
# ask for. The instant, solar geometry and lens that used to sit here were
# detail a site plan's reader does not need.
# Cut to what the figure cannot do without: the source of the ring sizes, one
# word that the siting is indicative, and the attributions the data licences
# require. Everything else that used to sit here -- the instant, the solar
# geometry, the lens, the acceleration-turns caveat -- was detail a site plan's
# reader does not need, and it crowded the frame.
PROVENANCE = "Ring sizes: IMCC parameter list, 2023. Siting indicative."
CREDIT = ("\u00a9 OpenStreetMap contributors (ODbL) \u00b7 AWS Terrain Tiles \u00b7 "
          "star field NASA/GSFC, Gaia DR2 (ESA/Gaia/DPAC)")

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
FOOTER_LINES = 2             # reserved; the block wraps into at most this many
FOOTER_RULE = FOOTER_BOTTOM + FOOTER_LINES * FOOTER_LEAD + 0.014
# 0.130 tall when it held the sequence ribbon above the key; the ribbon is gone,
# so the band is the key and the scale bar and nothing else. The 0.055 this
# releases goes to the callout ladder, which needs it for the wrapped copy.
LEGEND_BAND = (FOOTER_RULE + 0.014, FOOTER_RULE + 0.075)
TITLE_FLOOR = 0.800          # nothing else goes above this on the left
# The columns need to be wide enough for the longest metric string; an earlier
# 0.098 gave 157 px for a 187 px string, so six blocks hung past the margin.
# Set from measured type, not guessed: with the trimmed copy the widest block
# is the label "6 Detector halls" at 170 px = 0.106 of a 1600 px frame, so a
# 0.120 offset holds every string with a pad to spare and leaves the drawing the
# middle 64 % of the width instead of 48 %. That corridor is what lets the site
# outline stay clear of the text while the camera zooms out rather than in.
# (0.115 was tried first, from the widest *metric* at 168 px -- the checker's
# margin test caught the label overrunning it by 2 px.)
# 0.135, not 0.120, and the extra 0.015 is measured rather than tasteful. The
# purpose lines wrap to the column, so a narrower column is a taller callout:
# at 0.120 the eight blocks overran the title block and no balanced split fit.
# Wider is not free either -- at 0.165 the left gutter reaches the Main
# Injector's own anchor, which needs an 86 degree diagonal to get out, and one
# leader like that sets the angle for all of them. 0.135 is where the copy fits
# and the steepest leader still only needs 55 degrees.
COL_X = (MARGIN + 0.135, 1.0 - MARGIN - 0.135)
# 0.025 of clearance above the legend band, not 0.045. The band was 0.130 tall
# when it carried the sequence ribbon and the 0.045 was measured against that;
# with the band down to 0.075 and holding only the key and the scale bar, its
# stated top already clears everything drawn inside it.
LABEL_FLOOR = LEGEND_BAND[1] + 0.025   # the ladder may not reach into the legend
# How far above its anchor each label sits. A rung level with its anchor gives a
# flat leader and so no dog-leg at all; this is what puts the angle back. It is
# a uniform shift, so it cannot reintroduce the crossings that an unrelated
# ladder height once caused -- those came from the *order* going non-monotonic.
LABEL_LIFT = 0.105


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
LEADERS: list[tuple[str, tuple, tuple, tuple]] = []   # (key, anchor, knee, end) in axes coords
# The diagonal angle every dog-leg in the figure shares, in degrees measured
# off the horizontal in *pixel* space, so it looks like one angle on screen
# whatever the frame's aspect. draw_features sets it from the geometry: the
# steepest leg any one callout needs, applied to all of them, which is the only
# way to hold a single angle without some legs having to be clamped steeper.
LEADER_ANGLE = [45.0]
LEADER_ANGLE_LIMITS = (30.0, 62.0)


def leader_angle_needed(ax_x, ax_y, lx, ly, width, height):
    """The diagonal angle this one callout needs if its knee is to reach the
    gutter without the leg overshooting the text edge."""
    run = abs(lx * width - ax_x * width) - GUTTER * width
    rise = abs(ly * height - ax_y * height)
    if run <= 1.0:
        return LEADER_ANGLE_LIMITS[1]
    return math.degrees(math.atan2(rise, run))


def leader_path(ax_x, ax_y, lx, ly, ha, width, height, text_edge):
    """The dog-leg polyline in pixels, or None where no line should be drawn.

    A fixed rise over a varying run gives every leader its own angle, which is
    what a ladder of them looked like: nine legs at nine slopes reading as
    nine unrelated marks. Here the angle is fixed and the *knee* floats -- the
    diagonal leaves the anchor at LEADER_ANGLE and turns wherever it reaches
    the rung, then runs level into the text. Parallel diagonals also cannot
    cross each other, which the shared-gutter version could not promise.
    """
    towards = 1.0 if ha == "left" else -1.0
    if (towards < 0 and ax_x < text_edge) or (towards > 0 and ax_x > text_edge):
        return None                                # behind its own label
    ax_px, ay_px = ax_x * width, ax_y * height
    lx_px, ly_px = lx * width, ly * height
    run = abs(ly_px - ay_px) / math.tan(math.radians(LEADER_ANGLE[0]))
    knee = ax_px + towards * run
    knee = min(knee, lx_px) if towards > 0 else max(knee, lx_px)
    return [(ax_px, ay_px), (knee, ly_px), (lx_px, ly_px)]
# Measured extent and colour of every block drawn, in pixels, for
# check_figure.py --blocks. The layout checker here can only see the layout; it
# cannot know what the render puts *behind* a label, so the WCAG reading has to
# be taken on pixels by the other tool. It documented a --blocks input from the
# start and nothing ever wrote the file, so that reading had never been taken.
BLOCKS: list[dict] = []


def leader(ax, ax_x, ax_y, lx, ly, colour, s, width, height, *, ha="left", key="",
           dashed=False, text_edge=None, zorder=6):
    """Dog-leg leader: a diagonal off the anchor at the figure's one angle,
    then a short level run into the text block.

    The line stays one neutral grey rather than the category hue: where a cyan
    leader crossed a cyan tunnel the reader could not tell pointer from
    subject. But it is drawn at full weight -- a dog-leg has to be traceable
    from label to subject at a glance, and an earlier pass had de-weighted it
    to the point of invisibility.

    Where the anchor lies behind its own label block no line is drawn: the
    label is already beside its subject, so a connector would only cross the
    type it belongs to. The survey mark carries the association instead, which
    is what direct labelling means.
    """
    text_edge = lx if text_edge is None else text_edge
    path = leader_path(ax_x, ax_y, lx, ly, ha, width, height, text_edge)
    if path is not None:
        xs = [q[0] / width for q in path]
        ys = [q[1] / height for q in path]
        ax.plot(xs, ys, transform=ax.transAxes, color=LEADER,
                alpha=0.92, lw=1.3 * s, solid_capstyle="round", solid_joinstyle="miter",
                dashes=(4, 3) if dashed else (None, None), zorder=zorder,
                path_effects=[patheffects.withStroke(linewidth=3.0 * s, foreground="#05070Bcc")])
        if abs(xs[1] - xs[0]) > 1e-4:
            # a dot at the corner, so the turn reads as deliberate
            ax.plot([xs[1]], [ys[1]], transform=ax.transAxes, marker="o", ms=2.4 * s,
                    mfc=LEADER, mec="none", alpha=0.95, zorder=zorder + 1)
        LEADERS.append((key, (xs[0], ys[0]), (xs[1], ys[1]), (xs[2], ys[2])))
    # the survey mark keeps the category colour: it is the one place the leader
    # touches its subject, so it is where the classification belongs
    if dashed:
        ax.plot([ax_x], [ax_y], transform=ax.transAxes, marker="o", ms=4.4 * s,
                mfc="none", mec=colour, mew=1.1 * s, zorder=zorder + 2)
    else:
        ax.plot([ax_x], [ax_y], transform=ax.transAxes, marker="o", ms=3.6 * s,
                mfc=colour, mec="none", zorder=zorder + 2)
    ax.plot([ax_x], [ax_y], transform=ax.transAxes, marker="o", ms=9.5 * s,
            mfc="none", mec=colour, mew=0.8 * s, alpha=0.45, zorder=zorder + 1)


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
         shadow=True, alpha=1.0, role="text", group=None, linespacing=None,
         halo=1.0, halo_alpha="99"):
    """One text block, with a halo proportional to the type it protects.

    The halo used to be a fixed 2.2 px at every size, which is 8.5 % of the em
    on the title and 23 % on a metric -- so the small type sat in a black furry
    box with its counters clogged, and the accent colour never reached its
    specified value. Scaling it with the size makes it one policy rather than
    three accidental ones.
    """
    t = ax.text(x, y, body, transform=ax.transAxes, ha=ha, va=va,
                fontproperties=fp, fontsize=size * s, color=colour, zorder=zorder,
                alpha=alpha, linespacing=linespacing)
    # `group` marks blocks that are meant to sit close -- the lines of one
    # paragraph, a name and its metric -- so the checker does not read
    # deliberate leading as a collision. Everything else needs full clearance.
    DRAWN.append((role, group or role, t))
    if shadow:
        # `halo` multiplies both the cap and the ratio, for the two blocks that
        # have no ground but the sky: with the panels gone the title and deck
        # sit on the brightest part of the frame, and at the standard halo they
        # measured WCAG 3.3 and 2.3 against it.
        lw = min(0.13 * halo * size * s, 2.6 * halo * s)
        t.set_path_effects([patheffects.withStroke(
            linewidth=lw, foreground="#05070B" + halo_alpha)])
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


def draw_title(ax, F, s, height, copy=None):
    """Title block. No institutional eyebrow -- it implied an official document,
    and dropping it lets the title hold the top of the frame on its own.

    The block sits high enough to clear the horizon glow. The deck used to be
    set at 12 pt in a cool grey, which made the line carrying the figure's
    actual claim the least legible text on the page; it is now the
    second-largest thing in the frame.

    Nothing here sits on a panel any more. Every block in this figure is set
    directly on the render and separated by its own halo, so the sky reaches
    the top of the frame with its star field intact.
    """
    copy = copy or TITLE
    x = MARGIN
    text(ax, x, 0.950, copy["title"], F["sans"], TYPE["title"], INK, s, va="top",
         role="title")
    # a measured tick, not an orphaned underline: the old 99 px rule under a
    # 560 px title read as neither. Neutral, so the title block does not enrol
    # itself in the categorical colour scale.
    ax.plot([x, x + 0.150], [0.874, 0.874], transform=ax.transAxes, color=INK,
            lw=1.6 * s, alpha=0.35, solid_capstyle="butt", zorder=8)
    # 1.32 only here, and the parameter defaults to None everywhere else for a
    # reason worth recording: passing linespacing *at all* switches matplotlib
    # from measured glyph extents to the nominal line box -- 12.50 px against
    # 10.50 for the same string -- so setting even 1.20, matplotlib's own
    # figure, grew all 46 blocks by 2 px and broke three deliberate name/metric
    # pairs into reported collisions.
    # 0.812, not 0.852. With the panels gone the deck's ground is whatever the
    # render puts behind it, and at 0.852 that was the brightest band of the
    # twilight sky: light type on light sky, measured at WCAG 2.30. No ink and
    # no halo can fix that -- reaching 4.5 against a ground that bright would
    # need a relative luminance above 1.0 -- so the block moves instead, far
    # enough to clear the horizon glow and sit on the land. Measured across the
    # move: 2.30 at 0.852, 4.39 at 0.832, 6.21 at 0.817, 6.60 here. Further
    # down the ladder starts to overrun.
    t = text(ax, x, 0.812, copy["deck"], F["sans"], TYPE["deck"], "#D8D3C9", s, va="top",
             role="deck", linespacing=1.32)
    # Return where the block actually ends, rather than leaving the ladder to
    # trust a typed constant. TITLE_FLOOR was 0.800 and the deck's measured
    # bottom is 0.769: the moment the deck took a second line the constant was
    # wrong, and the first thing the ladder did was put a label through it.
    r = RENDERER[0]
    return (t.get_window_extent(r).y0 / height) if r is not None else TITLE_FLOOR


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
                 va="center", role="legend:key", group="key")
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
             va="center", role=role, group="footer")
        y -= FOOTER_LEAD


def draw_scale(ax, F, s, anno, width, height):
    """A single horizontal graphic scale, right-aligned with the margin.

    One caveat belongs on the record. This projection is anisotropic: at the
    site centre 1 km east projects to 206 px and 1 km north to 95 px, a ratio
    of 2.29:1. A horizontal bar is therefore exact for east-west distances and
    reads 2.3x short for north-south ones, so a reader who measures the
    collider ring's north-south extent against it will get 1.4 km where the
    ring is 3.2 km. The two-legged L that used to sit here made that visible
    and was removed by request; the bar states what it measures and no more.
    """
    sc = anno.get("scale", {})
    px_per_km = sc.get("px_per_km_at_reference", 0.0)
    if px_per_km <= 0:
        return
    # Pick the round distance that fits the band rather than assuming one: this
    # camera projects 1 km to 206 px, so an assumed 2 km would be a 412 px bar.
    # 1 km first, because that is the distance asked for; the others are only
    # fallbacks for a camera where 1 km would be a bar too long to fit the band
    # or too short to tick.
    km = 1.0
    for cand in (1.0, 2.0, 5.0, 0.5):
        if 0.04 <= cand * px_per_km / width <= 0.20:
            km = cand
            break
    east = km * px_per_km / width

    # one shared right edge with the margin and the footer rule: three
    # near-identical right edges 20-140 px apart read as sloppiness
    x1 = 1.0 - MARGIN
    x0 = x1 - east
    y0 = LEGEND_BAND[0] + 0.010

    ax.plot([x0, x1], [y0, y0], transform=ax.transAxes, color=INK, lw=1.9 * s,
            solid_capstyle="butt", zorder=8)
    for f in (0.0, 0.5, 1.0):
        tx = x0 + east * f
        ax.plot([tx, tx], [y0, y0 - 0.012], transform=ax.transAxes, color=INK,
                lw=1.4 * s, zorder=8)
    text(ax, x1, y0 - 0.032, f"{km:g}\u202fkm", F["sans_med"], TYPE["meta"], INK, s,
         ha="right", va="center", role="legend:scale", group="scale")


# All three set from measured type, not chosen by eye. At 800x533 a name box
# measures 13.54 px and a purpose line 10.83 px, so a 0.019 lead put two
# purpose lines 10.1 px apart -- inside each other -- and a 0.016 gap overlapped
# the name it sat under. These clear both with a pixel or two to spare and hold
# at any width, since the type scales with the frame and these are fractions
# of it.
# Each is its measured minimum plus a little: a name box is 0.0254 of height
# and a purpose line 0.0203, the checker's pad is 0.0009 within a callout and
# 0.0056 between two, so the floors are 0.0213 for the lead, 0.0239 above and
# 0.0285 below. Moving the deck down for its contrast cost the ladder 0.040 of
# ceiling, and this is where it came back from -- there was 0.011 per callout
# sitting in gaps that were rounder than they needed to be.
SUB_LEAD = 0.0235        # leading within a callout's purpose block
SUB_GAP_ABOVE = 0.026    # name baseline to the first purpose line
SUB_GAP_BELOW = 0.030    # last purpose line to the next callout's name


def place_rungs_in_order(anchors, gaps, *, floor=None, ceil=None, lift=None):
    """Rungs for a ladder whose reading order is already fixed, top to bottom.

    place_rungs() chose the order itself, from anchor height. This one is given
    the order -- beam sequence -- and only has to find heights for it: each rung
    starts level with its own anchor plus the common lift, is pushed down as far
    as the block above it requires, and the whole ladder is then slid to fit
    between floor and ceiling. `gaps[k]` is the clearance the k-th block needs
    below its own baseline, so a four-line purpose gets four lines of room and a
    one-line purpose does not pay for them.
    """
    ceil = TITLE_FLOOR - 0.030 if ceil is None else ceil
    floor = LABEL_FLOOR if floor is None else floor
    lift = LABEL_LIFT if lift is None else lift
    n = len(anchors)
    if n == 0:
        return []
    # The last rung is a baseline with a whole block hanging under it, so the
    # floor it has to clear is its own depth above LABEL_FLOOR. Treating the
    # baseline as the floor is what put two purpose blocks through the legend.
    floor = floor + gaps[-1] - SUB_GAP_BELOW
    ys = [min(a + lift, ceil) for a in anchors]
    for k in range(1, n):                       # honour the order, push downward
        ys[k] = min(ys[k], ys[k - 1] - gaps[k - 1])
    below = ys[-1] - floor                      # the last block's own room
    if below < 0:                               # ran out of frame: slide up
        ys = [y - below for y in ys]
    # The gaps are what the type measures, so they are never scaled down to
    # make a column fit -- an earlier version did, and squeezing four wrapped
    # purpose blocks into three blocks' worth of room simply overlapped them.
    # The deficit is reported instead, and the column search treats it the way
    # it treats a crossing: as a layout to reject.
    return ys, max(0.0, ys[0] - ceil)


def _angle_for(placed, feats, width, height):
    """One angle for every dog-leg in the figure, taken from the callout that
    needs the steepest leg. Any shallower choice would leave that one to be
    clamped, and a single clamped leader is exactly the odd angle out that
    having a fixed angle is meant to prevent."""
    need = [leader_angle_needed(feats[keys[0]]["x"], 1.0 - feats[keys[0]]["y"],
                                lx, ly - 0.008, width, height)
            for _g, keys, lx, ly, _ha, _tx, _lines, _d in placed]
    lo, hi = LEADER_ANGLE_LIMITS
    return min(max(max(need) if need else 45.0, lo), hi)


def _count_crossings(placed, feats, width, height, angle):
    """Leader-on-leader crossings for a candidate layout."""
    was, LEADER_ANGLE[0] = LEADER_ANGLE[0], angle
    try:
        paths = []
        for _g, keys, lx, ly, ha, tx, _lines, _d in placed:
            f = feats[keys[0]]
            pth = leader_path(f["x"], 1.0 - f["y"], lx, ly - 0.008, ha,
                              width, height, tx)
            if pth:
                paths.append([(pth[0], pth[1]), (pth[1], pth[2])])
    finally:
        LEADER_ANGLE[0] = was
    hits = 0
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if any(_crosses(a, b, c, d) for a, b in paths[i] for c, d in paths[j]):
                hits += 1
    return hits


def draw_features(ax, F, s, anno, layout_name, width, height, *, ceil=None):
    """Place the callout ladders: side from the geometry, order from the beam.

    Two things are settled separately here, and conflating them is what earlier
    versions got wrong. *Which column* a callout goes in is a geometric question
    -- a label belongs on the side its own anchor leans toward, or its leader has
    to cross the drawing to reach it. *Where in the column* it goes is an
    editorial one: the figure's subject is a chain, so the ladder reads in beam
    order, 1 at the top down to 6, with the context items that carry no stage
    below them.

    Ordering by stage rather than by anchor height means a leader can now run
    against the grain of the ladder. That is a real cost, paid deliberately:
    the sequence ribbon that used to carry the beam order along the bottom of
    the frame is gone, so the ladder is the only place the order can live.
    check_collisions still reports any crossing it produces.
    """
    groups = []
    feats = anno.get("features", {})
    for g in LAYOUTS.get(layout_name, []):
        keys = [k for k in g["keys"] if k in feats and feats[k].get("on_screen")]
        if keys:
            groups.append((g, keys))

    # Which column, by search rather than by rule.
    #
    # Ranking the anchors by x and halving gives each callout the side it leans
    # toward, and that is the right instinct: it keeps every leader short and
    # off the drawing. But it takes no account of the reading order, and with
    # the ladder ordered by beam stage the two requirements collide. In this
    # camera they collide unavoidably: the detector halls are the highest thing
    # on screen and the *last* stage of the beam, so any column holding them
    # reads top-to-bottom in one order and bottom-to-top in the other, and a
    # leader has to cross. Partitioning the five stages into two columns that
    # are each monotonic in both is provably impossible here -- stage 5 has the
    # highest anchor of all, so it would have to come first in its column.
    #
    # So the split is chosen instead of derived: every balanced two-way
    # assignment is laid out, its leaders counted for crossings, and the one
    # with none wins -- breaking ties on total leader length, which recovers
    # the "short leaders on the near side" property the x-rank split had.
    # 2^n assignments at n = 8 is 256 layouts of pure arithmetic.
    def _layout(assign):
        out, total, over, first = [], 0.0, 0.0, {}
        for side in ("left", "right"):
            items = [g for g, a in zip(groups, assign) if a == side]
            if not items:
                return None, 1e9, 1e9, 1e9, 1e9
            lx = COL_X[0] if side == "left" else COL_X[1]
            ha = "right" if side == "left" else "left"
            tx = lx + (0.010 if ha == "left" else -0.010)
            avail = (tx - MARGIN) if ha == "right" else (1.0 - MARGIN - tx)
            # beam order, then the unstaged context below it
            items = sorted(items, key=lambda t: (t[0].get("stage") or 99,
                                                 -(1.0 - feats[t[1][0]]["y"])))
            # Wrapped in design space -- at s = 1 against SCALE_BASE -- not at
            # the output scale. A glyph's advance is not exactly linear in point
            # size once hinting and rounding are in it, so measuring at the
            # output size broke a line in a different place at 800 px than at
            # 1600, which changed the line counts, which changed the gaps,
            # which made the search choose a different column split and refuse
            # the figure at one resolution while passing it at the other.
            wrapped = [fit_lines(ax, g.get("purpose") or "", F["sans"],
                                 TYPE["metric"], 1.0, avail, SCALE_BASE)
                       for g, _ in items]
            gaps = [SUB_GAP_ABOVE + len(w) * SUB_LEAD + SUB_GAP_BELOW for w in wrapped]
            anchors = [1.0 - feats[keys[0]]["y"] for _, keys in items]
            ys, deficit = place_rungs_in_order(anchors, gaps, ceil=ceil)
            over += deficit
            for (g, keys), ly, lines in zip(items, ys, wrapped):
                out.append((g, keys, lx, ly, ha, tx, lines,
                            feats[keys[0]].get("depth_m", 0.0)))
                total += abs(lx - feats[keys[0]]["x"]) + abs(ly - (1.0 - feats[keys[0]]["y"]))
            first[side] = min([g.get("stage") or 99 for g, _ in items])
        # The ladder reads in beam order down a column, but a reader meets the
        # left column first, so the beam has to start there: an otherwise good
        # split put stages 4 and 5 on the left and 1 to 3 on the right, which
        # reads 4, 5, 1, 2, 3.
        backwards = 1 if first["left"] > first["right"] else 0
        return out, over, _angle_for(out, feats, width, height), backwards, total

    def over_cost(v):
        # 0.002, not 0.004. The tolerance is spent out of the clearance between
        # the top rung and the deck above it, and at 0.004 it spent more than
        # there was: the label's own box reaches 0.0127 above its rung, so a
        # 0.004 overrun left 0.0053 where the collision pad wants 0.0056 -- and
        # the 1600 px figure, which wraps to slightly different line counts and
        # so chooses a different split, landed exactly there.
        return round(max(0.0, v - 0.002), 4)

    n = len(groups)
    lo_n, hi_n = max(1, n // 2 - 1), (n + 1) // 2 + 1
    best = (None, 1e9, 1e9, 1e9, 1e9, None)
    x_rank = sorted(range(n), key=lambda i: feats[groups[i][1][0]]["x"])
    default = ["left" if x_rank.index(i) < (n + 1) // 2 else "right" for i in range(n)]
    for bits in range(1 << n):
        assign = ["left" if bits >> i & 1 else "right" for i in range(n)]
        if not lo_n <= assign.count("left") <= hi_n:
            continue
        out, over, angle, backwards, total = _layout(assign)
        if out is None:
            continue
        # Two things are being minimised, in this order.
        #
        # Overflow first, and as a magnitude rather than as a flag: as a flag it
        # collapsed a 0.0007 overrun and a 0.136 one into the same bucket, so
        # when every split overflowed a little the search ranked by the second
        # key and picked one that put four purpose lines through the deck. A
        # sub-pixel overrun is free; past that nothing outranks it, because
        # overlapping type is the one fault with no reading at all.
        #
        # Then the steepest angle any callout needs. The angle constraint is
        # one-sided -- a leader can always take a *steeper* diagonal than it
        # needs, since that only moves its knee nearer the anchor -- so a single
        # figure-wide angle equal to the steepest requirement fits every leader
        # with nothing clamped. Minimising that maximum is therefore how the
        # figure gets one shallow, consistent kink instead of one leader at 86
        # degrees dragging the rest up with it.
        #
        # Crossings are deliberately not in the key. They were, and optimising
        # them cost either the beam order or the shared angle; a crossed pair of
        # leaders is legible and an inconsistent set of kinks is not.
        if ((over_cost(over), round(angle, 1), backwards, total)
                < (over_cost(best[1]), round(best[2], 1), best[3], best[4])):
            best = (out, over, angle, backwards, total, assign)
    if best[0] is None:                              # no balanced split at all
        best = (*_layout(default), default)
    placed, over, angle = best[0], best[1], best[2]
    if over > 1e-6:
        print(f"[annotate] no column split fits: the best overruns the title "
              f"block by {over:.3f} of frame height. Shorten a purpose line.")
    LEADER_ANGLE[0] = angle
    if angle >= LEADER_ANGLE_LIMITS[1] - 1e-6:
        print(f"[annotate] WARNING: the shared dog-leg angle hit its "
              f"{LEADER_ANGLE_LIMITS[1]:.0f} deg ceiling, so at least one leader "
              f"is clamped steeper than the rest. Widen the drawing corridor or "
              f"shorten the ladder.")

    # far to near, so a nearer callout draws over a farther one
    for g, keys, lx, ly, ha, tx, lines, _ in sorted(placed, key=lambda t: -t[7]):
        prim = feats[keys[0]]
        accent = g.get("accent", prim.get("accent", "neutral"))
        colour = ACCENT.get(accent, ACCENT["neutral"])
        dashed = accent in DASHED
        ax_x, ax_y = prim["x"], 1.0 - prim["y"]
        leader(ax, ax_x, ax_y, lx, ly - 0.008, colour, s, width, height, ha=ha,
               key=keys[0], dashed=dashed, text_edge=tx)
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
        text(ax, tx, ly, label, F["sans_semi"], TYPE["label"], colour, s, ha=ha,
             va="center", role=f"label:{keys[0]}", group=f"callout:{keys[0]}")
        y = ly - SUB_GAP_ABOVE
        for line in lines:
            text(ax, tx, y, line, F["sans"], TYPE["metric"], METRIC_INK, s,
                 ha=ha, va="center", role=f"metric:{keys[0]}", group=f"callout:{keys[0]}")
            y -= SUB_LEAD

    wanted = {k for g in LAYOUTS.get(layout_name, []) for k in g["keys"]}
    missing = [k for k in wanted if k not in feats or not feats[k].get("on_screen")]
    if missing:
        print(f"[annotate] not drawn (off screen or absent): {', '.join(sorted(missing))}")
    print(f"[annotate] {len(placed)} callouts, dog-legs at {LEADER_ANGLE[0]:.1f} deg")
    return len(placed)


def _crosses(p1, p2, p3, p4):
    """True if segment p1-p2 properly crosses p3-p4 (pixel coordinates)."""
    def o(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return 0 if abs(v) < 1e-9 else (1 if v > 0 else -1)
    d1, d2 = o(p3, p4, p1), o(p3, p4, p2)
    d3, d4 = o(p1, p2, p3), o(p1, p2, p4)
    return d1 != d2 and d3 != d4 and 0 not in (d1, d2, d3, d4)


def _seg_hits_box(p, q, x0, y0, x1, y1) -> bool:
    """Liang-Barsky: does the segment p-q intersect the axis-aligned box?"""
    dx, dy = q[0] - p[0], q[1] - p[1]
    t0, t1 = 0.0, 1.0
    for num, den in ((x0 - p[0], dx), (p[0] - x1, -dx), (y0 - p[1], dy), (p[1] - y1, -dy)):
        if den == 0:
            if num > 0:
                return False
            continue
        t = num / den
        if den > 0:
            if t > t1:
                return False
            t0 = max(t0, t)
        else:
            if t < t0:
                return False
            t1 = min(t1, t)
    return t0 <= t1


def check_collisions(fig, ax, width, height, *, s=1.0, pad_px=6.0):
    """Measure the layout and report everything that is actually wrong with it.

    The previous version tested pairwise text overlap and a 2 px canvas clip,
    and printed "spacing OK" for a figure in which six blocks hung up to 48 px
    past the left margin and seven pairs of leaders crossed. Passing its own QA
    while breaking its own margin is the failure mode this function exists to
    prevent, so it now checks the things that were wrong:

      * text against text, and against the reserved bands
      * every block against the margin, not just the trim
      * leader against leader, as real segment intersections
      * leader against every text box, which is how a run through its own
        metric line went unnoticed
      * anything running off the canvas

    A guard that measures the wrong quantity is worse than no guard, because it
    licenses the belief that the layout has been checked.

    Every tolerance scales with `s`, the frame's width over the 1600 px
    reference, because everything it measures does. With the pads fixed in
    absolute pixels the same layout passed at 1600 px and reported fifteen
    faults at 800 px -- fourteen of them label/metric pairs that are deliberately
    close and exempted, whose gap in pixels halves with the frame while a 1 px
    exemption does not. That would have failed --strict on every render at any
    other width, the CI matrix at 960 px included.
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
        # `size` is the *design* size, not size*s: s is only the output
        # resolution, so a title is display type whether the figure is written
        # at 800 px or 4K. check_figure.py needs it to apply WCAG's large-text
        # threshold to the blocks the standard actually grants it to.
        BLOCKS.append({"role": role, "group": grp,
                       "x0": bb.x0, "y0": bb.y0, "x1": bb.x1, "y1": bb.y1,
                       "size": t.get_fontsize() / max(s, 1e-9),
                       "weight": str(t.get_fontproperties().get_weight()),
                       "rgb": [round(255.0 * c) for c in mcolors.to_rgb(t.get_color())]})

    def overlap(a, b, pad):
        return not (a[3] + pad <= b[1] or b[3] + pad <= a[1]
                    or a[4] + pad <= b[2] or b[4] + pad <= a[2])

    hits = []
    notes = []          # counted and reported, but not grounds for refusing
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            pad = (1.0 if a[5] == b[5] else pad_px) * s   # same group: close is intended
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
            if overlap((role, x0, y0, x1, y1), z, 2.0 * s):
                hits.append((role, zname, "text/zone"))

    # the margin is a specification, not an outcome: a flush edge that some
    # lines cross is not an edge
    ml, mr = MARGIN * width, (1.0 - MARGIN) * width
    for role, x0, y0, x1, y1, _grp in boxes:
        if x0 < ml - 1.0 * s:
            hits.append((role, f"left margin by {ml - x0:.0f} px", "margin"))
        if x1 > mr + 1.0 * s:
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
                        # A note, not a fault. Leader-on-leader crossings were
                        # treated as faults, and avoiding them cost either the
                        # beam order of the ladder or the one shared kink angle
                        # -- both of which carry more meaning than a crossing
                        # costs. A crossed pair is still worth counting, so it
                        # is reported and does not fail --strict. A leader
                        # through *type* is a different matter and stays a
                        # fault, below.
                        notes.append((f"leader:{k1}", f"leader:{k2}", "cross"))
                        break
                else:
                    continue
                break

    # leader against text: the case that slipped through. A run terminating at
    # the metric's baseline is right until the anchor sits inboard of its own
    # column, at which point the run crosses the label it belongs to -- and the
    # checker passed it, because it had no test for a line meeting a box.
    for key, a, k, e in LEADERS:
        segs = [((a[0] * width, a[1] * height), (k[0] * width, k[1] * height)),
                ((k[0] * width, k[1] * height), (e[0] * width, e[1] * height))]
        for role, x0, y0, x1, y1, _grp in boxes:
            if role.split(":", 1)[-1] == key:
                pad = 1.0 * s      # its own label: touching the edge is the point
            else:
                pad = 2.0 * s
            bx0, by0, bx1, by1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
            for p, q in segs:
                # segment against an axis-aligned box, by clipping
                if _seg_hits_box(p, q, bx0, by0, bx1, by1):
                    hits.append((f"leader:{key}", role, "leader/text"))
                    break
            else:
                continue
            break

    # anything running off the canvas
    for role, x0, y0, x1, y1, _grp in boxes:
        if x0 < 2 * s or y0 < 2 * s or x1 > width - 2 * s or y1 > height - 2 * s:
            hits.append((role, "canvas edge", "clipped"))

    if hits:
        print(f"[annotate] {len(hits)} layout problem(s):")
        for a, b, kind in hits:
            print(f"    {kind:11s} {a}  <->  {b}")
    else:
        print(f"[annotate] layout OK: {len(boxes)} text blocks and {len(LEADERS)} leaders; "
              f"no overlaps within {pad_px * s:.1f} px, no margin breaks, "
              "no leader through type")
    if notes:
        print(f"[annotate] {len(notes)} accepted: "
              + ", ".join(f"{kind} {a}/{b}" for a, b, kind in notes))
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
        title_floor = draw_title(ax, F, s, height, copy=dict(
            title=args.title or TITLE["title"],
            deck=(args.deck.replace("\\n", "\n") if args.deck else TITLE["deck"]),
        ))
        draw_footer(ax, F, s, anno, width)
        draw_key(ax, F, s, width)
        draw_scale(ax, F, s, anno, width, height)
        # the ladder's top rung has to clear the deck by its own half-height
        # plus the collision pad, not by a round number
        draw_features(ax, F, s, anno, args.layout, width, height, ceil=title_floor - 0.030)

    hits = check_collisions(fig, ax, width, height, s=s) if not args.debug else []
    if hits and args.strict:
        print("[annotate] --strict: refusing to write a figure with spacing problems")
        return 1

    stem = args.out or os.path.splitext(args.image)[0] + ("_debug" if args.debug else "_annotated")
    if args.dump_blocks:
        # check_collisions measures the boxes, so this is written after it runs
        # rather than measuring everything twice.
        with open(args.dump_blocks, "w") as fh:
            json.dump(BLOCKS, fh, indent=1)
        print(f"[annotate] wrote {args.dump_blocks}: {len(BLOCKS)} block boxes")
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
    p.add_argument("--dump-blocks", default="", metavar="JSON",
                   help="write each block's measured box and colour for check_figure.py --blocks")
    return build(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
