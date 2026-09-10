"""Access to the baked site plan: real geometry for the scene to be built from.

`tools/fetch_geodata.py` pulls OpenStreetMap, `tools/fetch_dem.py` pulls a
terrain model, and `tools/bake_geodata.py` reduces both to
`assets/plan/site_plan.json` plus `assets/plan/dem.png`, which are committed.
This module is the read side: it turns those into the polygons, polylines and
ground heights the builders in `site.py` consume.

Everything is in the scene's metric frame -- x east, y north, metres, origin at
OSM's Wilson Hall -- so a builder never has to know about latitude, tiles or
projections.

OSM data is ODbL: the render's credit line carries the attribution.
"""
from __future__ import annotations

import json
import math
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PLAN_PATH = os.path.join(ROOT, "assets", "plan", "site_plan.json")

_plan: dict | None = None
_dem: tuple | None = None          # (rows, size, half_span, lo, span, datum)


def plan() -> dict:
    global _plan
    if _plan is None:
        if not os.path.exists(PLAN_PATH):
            raise FileNotFoundError(
                f"{PLAN_PATH} missing. Run tools/fetch_geodata.py then "
                "tools/bake_geodata.py (see their docstrings).")
        with open(PLAN_PATH) as fh:
            _plan = json.load(fh)
    return _plan


def available() -> bool:
    return os.path.exists(PLAN_PATH)


def boundary() -> list[tuple[float, float]]:
    """The site outline, closed, counter-clockwise or clockwise as mapped."""
    return [(p[0], p[1]) for p in plan()["boundary"]]


def boundary_area_km2() -> float:
    return plan()["boundary_area_km2"]


def derived(name: str) -> dict:
    """A fitted feature: `tevatron`, `main_injector` or `site_filler`."""
    return plan()["derived"][name]


def layer(name: str) -> list[dict]:
    return plan()["layers"].get(name, [])


def ways(name: str, *, closed: bool | None = None, min_pts=2, radius=None):
    """Polylines from a layer, optionally filtered to closed rings or a radius."""
    out = []
    for w in layer(name):
        pts = w["pts"]
        if len(pts) < min_pts:
            continue
        if closed is not None and bool(w.get("closed")) != closed:
            continue
        if radius is not None:
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            if math.hypot(cx, cy) > radius:
                continue
        out.append(w)
    return out


# --------------------------------------------------------------------------- #
# terrain
# --------------------------------------------------------------------------- #
def _load_dem():
    global _dem
    if _dem is not None:
        return _dem
    meta = plan().get("dem")
    if not meta:
        _dem = ()
        return _dem
    path = os.path.join(os.path.dirname(PLAN_PATH), meta["file"])
    if not os.path.exists(path):
        _dem = ()
        return _dem
    # Blender ships numpy; PIL it does not, so read the 16-bit PNG through
    # Blender's own image loader when PIL is absent
    rows = None
    try:
        from PIL import Image
        import numpy as np
        rows = np.asarray(Image.open(path)).astype("float32") / 65535.0
    except Exception:                                  # noqa: BLE001
        try:
            import bpy
            import numpy as np
            img = bpy.data.images.load(path, check_existing=True)
            w, h = img.size
            buf = np.empty(w * h * 4, dtype="float32")
            img.pixels.foreach_get(buf)
            # Blender gives bottom-up RGBA in scene-linear; the PNG is greyscale
            # so any channel will do, and sRGB decode does not apply to a
            # non-colour heightmap -- ask for it raw
            rows = buf.reshape(h, w, 4)[::-1, :, 0]
        except Exception:                              # noqa: BLE001
            _dem = ()
            return _dem
    lo, hi = meta["elev_min_m"], meta["elev_max_m"]
    _dem = (rows, rows.shape[0], meta["half_span_m"], lo, hi - lo,
            meta["elev_at_origin_m"])
    return _dem


def has_dem() -> bool:
    return bool(_load_dem())


def elev(x: float, y: float) -> float:
    """Ground height at (x, y) in scene metres, zero at the origin.

    Bilinear, because the alternative -- nearest neighbour on a 117 m grid --
    puts visible 117 m terraces across a landscape whose whole character is
    that it is smooth.
    """
    d = _load_dem()
    if not d:
        return 0.0
    rows, n, half, lo, span, datum = d
    u = (x + half) / (2.0 * half) * (n - 1)
    v = (half - y) / (2.0 * half) * (n - 1)            # image rows run north->south
    if not (0 <= u <= n - 1 and 0 <= v <= n - 1):
        return 0.0
    i0, j0 = int(u), int(v)
    i1, j1 = min(i0 + 1, n - 1), min(j0 + 1, n - 1)
    fu, fv = u - i0, v - j0
    a = rows[j0, i0] * (1 - fu) + rows[j0, i1] * fu
    b = rows[j1, i0] * (1 - fu) + rows[j1, i1] * fu
    return float((a * (1 - fv) + b * fv) * span + lo - datum)


def dem_grid(size: int, half_span: float):
    """(verts, faces) for a terrain grid of `size` x `size` over +/- half_span."""
    verts = []
    for j in range(size):
        for i in range(size):
            x = -half_span + 2.0 * half_span * i / (size - 1)
            y = -half_span + 2.0 * half_span * j / (size - 1)
            verts.append((x, y, elev(x, y)))
    faces = []
    for j in range(size - 1):
        for i in range(size - 1):
            a = j * size + i
            faces.append((a, a + 1, a + size + 1, a + size))
    return verts, faces


# --------------------------------------------------------------------------- #
# polygons
# --------------------------------------------------------------------------- #
def point_in(poly, x: float, y: float) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i][0], poly[i][1]
        x1, y1 = poly[(i + 1) % n][0], poly[(i + 1) % n][1]
        if (y0 > y) != (y1 > y):
            xi = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < xi:
                inside = not inside
    return inside


def bbox(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def scatter_in_polygons(polys, spacing: float, rng, *, jitter=0.45, limit=None):
    """Points on a jittered grid inside a set of polygons.

    A jittered grid rather than rejection-sampled uniform noise: real woodland
    reads as a canopy at fairly even density, and pure random placement clumps
    badly enough at these counts to look like noise instead of trees.
    """
    out = []
    for poly in polys:
        if len(poly) < 3:
            continue
        x0, y0, x1, y1 = bbox(poly)
        if (x1 - x0) < spacing or (y1 - y0) < spacing:
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            out.append((cx, cy))
            continue
        nx = max(int((x1 - x0) / spacing), 1)
        ny = max(int((y1 - y0) / spacing), 1)
        for i in range(nx + 1):
            for j in range(ny + 1):
                x = x0 + (x1 - x0) * i / max(nx, 1) + rng.uniform(-jitter, jitter) * spacing
                y = y0 + (y1 - y0) * j / max(ny, 1) + rng.uniform(-jitter, jitter) * spacing
                if point_in(poly, x, y):
                    out.append((x, y))
        if limit and len(out) >= limit:
            return out[:limit]
    return out[:limit] if limit else out
