"""Geometry shared by the metro (Fermilab-to-Chicago) view's fetcher, baker and builder.

No bpy here: tools/fetch_metro.py and tools/bake_metro.py import this from the
host Python, and scene/metro.py imports it inside Blender.

The frame. The site scene works in metres about OSM's Wilson Hall on a flat
equirectangular plane, which is exact to well under a metre across a 6 km
campus. The metro view spans 340 km, where that stops being true: at the top of
the frame the equirectangular x scale is 2.4 % off (2.4 km at 100 km east), and
the ground has dropped 2.4 km below the tangent plane. So this view uses the
*orthographic* projection onto the tangent plane at the same origin:

    x, y  = the east and north components of a point on the sphere
    z     = its height above the tangent plane (negative: the Earth curves away)

which is simply the point's true 3D position in the local east-north-up frame.
A mesh built on it is the curved Earth itself, not a map of it -- the horizon
comes out right because the geometry is right. Within the campus it agrees
with the site frame to a few centimetres, so the existing site geometry drops
into it unchanged.
"""
from __future__ import annotations

import math

R_EARTH = 6371000.0
# Same origin as assets/plan/site_plan.json: OSM's Wilson Hall centroid
LAT0, LON0 = 41.838327, -88.261661

# --- regions (scene metres: xmin, xmax, ymin, ymax) ----------------------------
# FAR covers the whole of camera A's footprint on the ground (solved by
# ray-marching the frame border onto the sphere: x -125..189 km, y -30..176 km)
# with a margin, so terrain, imagery and lights reach every pixel of the frame.
# NEAR is the half of the frame where a ground texel is resolvable: below about
# the frame's mid-height one render pixel is 24-60 m of ground; above it the
# depth foreshortening makes a pixel hundreds of metres deep and the coarse
# rasters are already finer than the pixel.
FAR = (-140000.0, 200000.0, -45000.0, 190000.0)
NEAR = (-30000.0, 85000.0, -35000.0, 50000.0)
# OUTER carries the ground to the horizon for the flight (scene/flight.py),
# whose 3:2 frame shows the horizon from 28 km: 597 km away at that height.
# At 1 km texels it is hazed to near-invisibility at those distances, but it is
# real ground with real city glow on it -- Milwaukee, Rockford, Gary, Grand
# Rapids -- rather than the edge of a mesh.
OUTER = (-480000.0, 520000.0, -140000.0, 580000.0)

# raster resolutions, metres per texel
RES = {
    "dem_near": 30.0, "dem_far": 240.0,
    "albedo_near": 15.0, "albedo_far": 120.0,
    "lights_near": 15.0, "lights_far": 60.0,
    "dem_outer": 2000.0, "albedo_outer": 1000.0, "lights_outer": 1000.0,
}

# --- places ------------------------------------------------------------------------
WILLIS_TOWER = (41.878876, -87.635915)
UCHICAGO_QUAD = (41.7897, -87.5997)       # Harper / the main quadrangle

# --- camera A ------------------------------------------------------------------------
# Solved with the pinhole sketch in the framing study and re-expressed in this
# frame: 28 km above the ground at its nadir, 59 km south of the line between
# the site and the Loop, looking north 25.4 deg down on a 26 mm lens (69.4 deg
# horizontal), lens-shifted down 1.35 % so the subject band sits a little below
# the frame's centre. At 3840 px wide it resolves 24 m/px at Fermilab and
# 26 m/px at the Loop.
CAM_A_XY = (26000.0, -62000.0)
CAM_A_ALT = 28000.0
CAM_A_TARGET_XY = (27500.0, -3000.0)
CAM_A_LENS = 26.0
CAM_A_SHIFT_Y = 0.0135


# --- projection -------------------------------------------------------------------------
_S0, _C0 = math.sin(math.radians(LAT0)), math.cos(math.radians(LAT0))


def to_xyz(lat, lon, h=0.0):
    """(lat, lon, height above the sphere) -> scene (x, y, z), metres. Scalar."""
    p, dl = math.radians(lat), math.radians(lon - LON0)
    sp, cp = math.sin(p), math.cos(p)
    r = R_EARTH + h
    x = r * cp * math.sin(dl)
    y = r * (_C0 * sp - _S0 * cp * math.cos(dl))
    u = r * (_S0 * sp + _C0 * cp * math.cos(dl)) - R_EARTH
    return x, y, u


def to_xy(lat, lon):
    x, y, _ = to_xyz(lat, lon)
    return x, y


def drop(x, y):
    """Height of the sphere below the tangent plane at plan position (x, y)."""
    r2 = x * x + y * y
    return math.sqrt(max(R_EARTH * R_EARTH - r2, 0.0)) - R_EARTH


def to_latlon(x, y):
    """Inverse orthographic: scene plan (x, y) -> (lat, lon). Scalar."""
    rho = math.hypot(x, y)
    if rho < 1e-6:
        return LAT0, LON0
    c = math.asin(min(rho / R_EARTH, 1.0))
    sc, cc = math.sin(c), math.cos(c)
    lat = math.asin(cc * _S0 + y * sc * _C0 / rho)
    lon = LON0 + math.degrees(math.atan2(x * sc, rho * cc * _C0 - y * sc * _S0))
    return math.degrees(lat), lon


def np_to_latlon(x, y):
    """Vectorised inverse (numpy arrays)."""
    import numpy as np
    rho = np.hypot(x, y)
    rho = np.where(rho < 1e-6, 1e-6, rho)
    c = np.arcsin(np.minimum(rho / R_EARTH, 1.0))
    sc, cc = np.sin(c), np.cos(c)
    lat = np.degrees(np.arcsin(cc * _S0 + y * sc * _C0 / rho))
    lon = LON0 + np.degrees(np.arctan2(x * sc, rho * cc * _C0 - y * sc * _S0))
    return lat, lon


def np_to_xy(lat, lon):
    import numpy as np
    p, dl = np.radians(lat), np.radians(np.asarray(lon) - LON0)
    sp, cp = np.sin(p), np.cos(p)
    x = R_EARTH * cp * np.sin(dl)
    y = R_EARTH * (_C0 * sp - _S0 * cp * np.cos(dl))
    return x, y


def region_latlon_bbox(region, pad_deg=0.02):
    """(lat_min, lat_max, lon_min, lon_max) enclosing a scene-metre region."""
    x0, x1, y0, y1 = region
    lats, lons = [], []
    n = 24
    for i in range(n + 1):
        for (x, y) in ((x0 + (x1 - x0) * i / n, y0), (x0 + (x1 - x0) * i / n, y1),
                       (x0, y0 + (y1 - y0) * i / n), (x1, y0 + (y1 - y0) * i / n)):
            la, lo = to_latlon(x, y)
            lats.append(la)
            lons.append(lo)
    return (min(lats) - pad_deg, max(lats) + pad_deg, min(lons) - pad_deg, max(lons) + pad_deg)


def grid_shape(region, res):
    x0, x1, y0, y1 = region
    return int(round((y1 - y0) / res)), int(round((x1 - x0) / res))    # rows, cols
