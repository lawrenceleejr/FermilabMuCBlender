"""Coordinate helpers shared by the fetch and build scripts. No bpy."""

import math

M_PER_DEG_LAT = 110574.0
M_PER_DEG_LON_EQ = 111320.0


def latlon_to_enu(lat, lon, origin_lat, origin_lon):
    """Equirectangular lat/lon -> local (east, north) meters.

    Error over a ~10 km box at 41.8 N is under ~10 m — fine for a
    visualization-scale scene.
    """
    x = (lon - origin_lon) * M_PER_DEG_LON_EQ * math.cos(math.radians(origin_lat))
    y = (lat - origin_lat) * M_PER_DEG_LAT
    return x, y


def enu_to_latlon(x, y, origin_lat, origin_lon):
    lat = origin_lat + y / M_PER_DEG_LAT
    lon = origin_lon + x / (M_PER_DEG_LON_EQ * math.cos(math.radians(origin_lat)))
    return lat, lon


# --- Web-Mercator slippy tiles ----------------------------------------------

def latlon_to_tile(lat, lon, zoom):
    """Lat/lon -> fractional slippy-map tile coordinates."""
    n = 2.0 ** zoom
    xt = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    yt = (1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n
    return xt, yt


def tile_to_latlon(xt, yt, zoom):
    """Fractional tile coordinates -> lat/lon of that point."""
    n = 2.0 ** zoom
    lon = xt / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * yt / n))))
    return lat, lon


def polygon_area(points):
    """Signed shoelace area of an (x, y) ring."""
    a = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return a / 2.0
