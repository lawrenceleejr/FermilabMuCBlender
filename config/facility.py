"""Geometry parameters for the Fermilab 10 TeV muon collider scene.

All positions are in a local ENU frame (meters): x = east, y = north,
z = up, origin at Wilson Hall ground level. Pure Python — importable both
by the fetch scripts (regular CPython) and by build_scene.py (Blender's
bundled Python). Must never import bpy.

Accelerator-complex numbers follow the Fermilab site-filler scenario in
the US Muon Collider white paper for the ESPPU (arXiv:2503.23695) and the
IMCC interim report (arXiv:2407.12450):

  - pre-acceleration (linac + RLAs) to ~63 GeV
  - RCS1  6.28 km  63 GeV -> 450 GeV    reuses the Tevatron tunnel (~8 m deep)
  - RCS2  10.5 km  450 GeV -> 1.725 TeV new intermediate tunnel
  - RCS3  16.5 km  1.725 -> 3.56 TeV    shared largest tunnel
  - RCS4  16.5 km  3.56 -> 5.0 TeV      same 16.5 km tunnel, stacked above RCS3
  - collider ring ~10 km, 10 TeV c.o.m., ~200 m deep in the
    Galena-Platteville dolomite (bottom of layer ~207 m below grade)

Rings are drawn as circles; the real machines are racetracks with straight
sections. Tunnel tube radii are visually exaggerated so the rings read
from aerial viewpoints.
"""

import math

# --- Georeference ---------------------------------------------------------
# Wilson Hall (centroid of the OSM footprint, way "Wilson Hall"); scene/ENU
# origin.
ORIGIN_LAT = 41.83841
ORIGIN_LON = -88.26148

# Fetch bbox (south, west, north, east) for terrain and OSM buildings.
FETCH_BBOX = (41.79, -88.31, 41.87, -88.20)

# --- Site layout -----------------------------------------------------------
# Tevatron ring center (41.8319 N, -88.2519 W) in ENU; Wilson Hall sits just
# outside the ring's northwest edge, ~1.07 km from the center.
TEVATRON_CENTER = (795.0, -720.0)

# The new deep rings are centered on the campus so the 16.5 km tunnel
# (radius 2626 m) fits the ~6800-acre site as well as possible: this is the
# max-clearance point of the OSM site boundary (clearance ~2518 m, so the
# largest ring pokes ~110 m past the fence at the tightest spot — faithful
# to the white paper's "almost exactly the size of the campus").
CAMPUS_CENTER = (1500.0, 400.0)

OKABE_ITO = {
    "orange":        (0.90, 0.62, 0.00),
    "sky_blue":      (0.34, 0.71, 0.91),
    "bluish_green":  (0.00, 0.62, 0.45),
    "yellow":        (0.94, 0.89, 0.26),
    "vermillion":    (0.84, 0.37, 0.00),
    "reddish_purple":(0.80, 0.47, 0.65),
    "blue":          (0.00, 0.45, 0.70),
}

# --- Rings -----------------------------------------------------------------
# Each ring: name/label, circumference (m), tunnel depth (m, negative down),
# center (E, N), display color (linear RGB), visual tube radius (m).
RINGS = [
    {
        "name": "RCS1",
        "label": "RCS1 (Tevatron tunnel) - 6.28 km, 63-450 GeV",
        "circumference": 6280.0,
        # nominal cut-and-cover Tevatron tunnel is ~8 m down; drawn deeper so
        # the exaggerated tube (12 m radius) stays below the terrain surface
        "depth": -25.0,
        "center": TEVATRON_CENTER,
        "color": OKABE_ITO["orange"],
        "tube_radius": 12.0,
    },
    {
        "name": "RCS2",
        "label": "RCS2 - 10.5 km, 0.45-1.725 TeV",
        "circumference": 10500.0,
        "depth": -100.0,
        "center": CAMPUS_CENTER,
        "color": OKABE_ITO["sky_blue"],
        "tube_radius": 15.0,
    },
    {
        "name": "RCS3",
        "label": "RCS3 - 16.5 km, 1.725-3.56 TeV",
        "circumference": 16500.0,
        "depth": -150.0,
        "center": CAMPUS_CENTER,
        "color": OKABE_ITO["bluish_green"],
        "tube_radius": 15.0,
    },
    {
        "name": "RCS4",
        "label": "RCS4 - 16.5 km, 3.56-5.0 TeV (shared tunnel)",
        "circumference": 16500.0,
        "depth": -150.0,  # same tunnel as RCS3, beamlines side by side
        "center": CAMPUS_CENTER,
        "color": OKABE_ITO["yellow"],
        "tube_radius": 15.0,
        "radius_offset": 45.0,  # visual: draw beside RCS3 in the shared tunnel
    },
    {
        "name": "Collider",
        "label": "Collider ring - 10 km, 10 TeV c.o.m.",
        "circumference": 10000.0,
        "depth": -200.0,
        "center": CAMPUS_CENTER,
        "color": OKABE_ITO["vermillion"],
        "tube_radius": 18.0,
    },
]


def ring_radius(ring):
    return ring["circumference"] / (2.0 * math.pi)


# --- Front end: proton driver / target / cooling / pre-acceleration --------
# Straight segments placed near the existing proton complex east of the
# Tevatron ring; gentle slope down toward RCS1 depth.
# Each: label, start (E, N, Z), end (E, N, Z), color, visual radius.
LINACS = [
    {
        "name": "TargetCooling",
        "label": "Target + cooling channel",
        "start": (900.0, 150.0, -10.0),
        "end": (900.0, -850.0, -10.0),
        "color": OKABE_ITO["reddish_purple"],
        "tube_radius": 10.0,
    },
    {
        "name": "PreAccelerator",
        "label": "Linac + RLAs (to 63 GeV)",
        "start": (900.0, -850.0, -10.0),
        "end": (1795.0, -720.0, -8.0),  # feeds into RCS1 (Tevatron ring, east point)
        "color": OKABE_ITO["blue"],
        "tube_radius": 10.0,
    },
]

TARGET_HALL = {
    "label": "Target hall",
    "center": (900.0, 200.0, -10.0),
    "size": (60.0, 60.0, 25.0),
    "color": OKABE_ITO["reddish_purple"],
}

# --- Collider IP caverns ----------------------------------------------------
# Two interaction points on opposite sides of the collider ring.
IP_CAVERNS = [
    {"name": "IP1", "azimuth_deg": 90.0},   # north side, near Wilson Hall
    {"name": "IP2", "azimuth_deg": 270.0},  # south side
]
IP_CAVERN_SIZE = (60.0, 35.0, 35.0)
IP_COLOR = OKABE_ITO["reddish_purple"]

# --- Access shafts ----------------------------------------------------------
# (ring name, azimuth degrees) -> cylinder from surface to ring depth.
SHAFTS = [
    ("RCS2", 0.0),
    ("RCS2", 180.0),
    ("RCS3", 45.0),
    ("RCS3", 225.0),
    ("Collider", 90.0),   # IP1 access
    ("Collider", 270.0),  # IP2 access
]
SHAFT_RADIUS = 8.0

# --- Buildings ---------------------------------------------------------------
WILSON_HALL_HEIGHT = 76.0
DEFAULT_BUILDING_HEIGHT = 5.0
METERS_PER_LEVEL = 3.2

# Fallback footprint (ENU meters around origin) used only if OSM lacks
# Wilson Hall: approximate twin-tower plan, ~70 x 60 m.
WILSON_HALL_FALLBACK = [
    (-35.0, -30.0), (35.0, -30.0), (35.0, -12.0), (10.0, -12.0),
    (10.0, 12.0), (35.0, 12.0), (35.0, 30.0), (-35.0, 30.0),
    (-35.0, 12.0), (-10.0, 12.0), (-10.0, -12.0), (-35.0, -12.0),
]

# Fallback site boundary: 5.2 km square centered on the campus center,
# used only if the OSM boundary relation cannot be fetched.
_B = 2600.0
SITE_BOUNDARY_FALLBACK = [
    (CAMPUS_CENTER[0] - _B, CAMPUS_CENTER[1] - _B),
    (CAMPUS_CENTER[0] + _B, CAMPUS_CENTER[1] - _B),
    (CAMPUS_CENTER[0] + _B, CAMPUS_CENTER[1] + _B),
    (CAMPUS_CENTER[0] - _B, CAMPUS_CENTER[1] + _B),
]

# --- Aerial imagery / surface overlays ---------------------------------------
IMAGERY_ZOOM = 16          # USGSImageryOnly max useful zoom here (~1.8 m/px)
IMAGERY_WIDTH = 4096       # output px E-W; N-S derived from heightmap aspect
IMAGERY_MAX_MB = 14.0
ONSITE_RADIUS = 2800.0     # "on campus" test radius around CAMPUS_CENTER
WATER_Z_OFFSET = 0.6       # water surface above mean sampled terrain (m)
WATER_MIN_AREA = 400.0     # m^2, drop micro-ponds

# --- Wilson Hall sculpted model ----------------------------------------------
# Twin cast-concrete towers sweeping inward toward the top, central atrium,
# crossover bridges from the 7th floor up; 16 stories, long axis ~38 deg east
# of north. Dimensions approximated from the OSM footprint (~97 x 104 m) and
# architectural references.
WILSON_HALL_MODEL = {
    "floors": 16,
    "floor_h": 4.6,            # => 73.6 m total
    "length": 97.0,            # long axis (local X before rotation)
    "half_width_base": 32.0,   # outer wall half-width at grade
    "half_width_top": 10.0,    # outer wall half-width at roof
    "sweep_exp": 2.4,          # outer-wall sweep exponent (higher = more
                               # dramatic flare at the base)
    "gap_half_base": 7.0,      # atrium half-gap at grade (14 m slot)
    "gap_half_min": 0.7,       # towers nearly touch at the top
    "rotation_deg": 38.0,      # long-axis bearing east of north
    "bridge_floors": (7, 9, 11, 13, 15),
    "bridge_width": 9.0,
}

# --- Building material palettes (linear RGB) ----------------------------------
BUILDING_PALETTE_OFFSITE = [
    (0.58, 0.55, 0.50),  # warm gray
    (0.52, 0.54, 0.57),  # cool gray
    (0.60, 0.56, 0.47),  # pale tan
    (0.48, 0.46, 0.43),  # gray-brown
]
BUILDING_PALETTE_ONSITE = [
    (0.55, 0.48, 0.40),  # warm concrete
    (0.42, 0.45, 0.50),  # cool concrete
    (0.55, 0.42, 0.30),  # tan brick
    (0.38, 0.40, 0.42),  # gray metal
]

# --- Geology (cutaway face shading) -----------------------------------------
GLACIAL_TILL_BOTTOM = -70.0     # tan glacial deposits above
DOLOMITE_BOTTOM = -207.0        # Galena-Platteville dolomite layer bottom
TERRAIN_SOLID_BOTTOM = -260.0   # bottom of the solid terrain block
