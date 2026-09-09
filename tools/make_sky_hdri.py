"""Build a night-sky HDRI as seen from Fermilab at a given instant.

Source: NASA/GSFC SVS "Deep Star Maps 2020" (ID 4851) -- an equirectangular
map of 1.7 billion stars (Gaia DR2, Hipparcos-2, Tycho-2) in ICRF/J2000
equatorial coordinates, centred on 0h RA with RA increasing to the left.
Public domain; credit NASA/GSFC SVS, Gaia DR2: ESA/Gaia/DPAC.

This script rotates that map into the local horizon frame (x east, y north,
z up -- the Blender world frame used by scene/) for Fermilab's latitude and
longitude at the requested UTC instant, adds a faint airglow floor and the
light-pollution domes of the Chicago area, and writes an equirectangular EXR
in Blender's environment-texture convention plus a tonemapped PNG preview.

Run inside Blender's Python (for EXR I/O):

    blender -b --python tools/make_sky_hdri.py -- \
        [--utc 2026-09-10T02:30] [--res 8k] [--out assets/hdri/fermilab_night_sky.exr]
        [--no-skyglow] [--scale S]

The default instant is 19:32 CDT on 9 Sep 2026 (00:32 UTC the 10th): the sun
is ~4.4 deg below the horizon in the WNW -- civil twilight, so there is still
a warm band along the western horizon and enough skylight to read the
landscape -- while the galactic centre stands 19 deg up in the south. The
script prints the solar position and writes it to a JSON sidecar beside the
EXR so the scene can place its twilight sun at the same instant as these
stars. Note that a real camera would not record the Milky Way this early in
twilight; the star field is added at full strength so it stays visible, and
`--star-scale` in build_scene.py dials that back.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
import urllib.request

import numpy as np

try:
    import bpy
except ImportError:  # pragma: no cover
    bpy = None

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STARMAP_DIR = os.path.join(ROOT, "assets", "starmap")
HDRI_DIR = os.path.join(ROOT, "assets", "hdri")
SVS = "https://svs.gsfc.nasa.gov/vis/a000000/a004800/a004851/starmap_2020_{res}.exr"

# Fermilab, Batavia IL (Wilson Hall)
LAT_DEG = 41.8319
LON_DEG = -88.2560  # east-positive

# Bright reference stars / objects (ICRS, degrees) for the verification table
REFS = {
    "Vega": (279.235, 38.784),
    "Deneb": (310.358, 45.280),
    "Altair": (297.696, 8.868),
    "Arcturus": (213.915, 19.182),
    "Polaris": (37.955, 89.264),
    "Capella": (79.172, 45.998),
    "Antares": (247.352, -26.432),
    "Fomalhaut": (344.413, -29.622),
    "Sgr A* (gal. centre)": (266.417, -29.008),
}

# Light pollution: (compass azimuth deg, relative strength, angular spread deg, scale height deg, colour)
# Distances/directions from Fermilab: Chicago ~55 km E, Naperville ~15 km ESE, Aurora ~10 km S,
# West Chicago ~6 km NE, Batavia/Geneva/St Charles 3-8 km NW-N, Elgin ~25 km NNE.
CITIES = [
    (88.0, 1.00, 28.0, 9.0, (1.00, 0.72, 0.50)),   # Chicago + inner suburbs: broad whitish-amber dome
    (108.0, 0.45, 12.0, 6.0, (1.00, 0.66, 0.40)),  # Naperville / Warrenville
    (185.0, 0.40, 12.0, 6.0, (1.00, 0.62, 0.35)),  # Aurora
    (42.0, 0.22, 9.0, 5.0, (1.00, 0.64, 0.38)),    # West Chicago / Winfield
    (345.0, 0.28, 16.0, 5.0, (1.00, 0.62, 0.35)),  # Batavia / Geneva / St Charles
    (20.0, 0.15, 10.0, 5.0, (1.00, 0.66, 0.40)),   # Elgin
]


# --------------------------------------------------------------------------- #
# astronomy
# --------------------------------------------------------------------------- #
def julian_date(t: dt.datetime) -> float:
    y, m = t.year, t.month
    d = t.day + (t.hour + t.minute / 60 + t.second / 3600) / 24
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5


def gmst_deg(jd: float) -> float:
    t = (jd - 2451545.0) / 36525.0
    g = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * t * t - t ** 3 / 38710000.0
    return g % 360.0


def horizon_basis(lat_deg: float, lst_deg: float) -> np.ndarray:
    """Rows = east, north, up unit vectors expressed in equatorial (x->RA 0, z->NCP)."""
    phi, th = math.radians(lat_deg), math.radians(lst_deg)
    up = np.array([math.cos(phi) * math.cos(th), math.cos(phi) * math.sin(th), math.sin(phi)])
    north = np.array([-math.sin(phi) * math.cos(th), -math.sin(phi) * math.sin(th), math.cos(phi)])
    east = np.array([-math.sin(th), math.cos(th), 0.0])
    return np.stack([east, north, up])


def sun_radec(jd: float) -> tuple[float, float]:
    """Apparent solar RA/Dec (deg), low-precision formulae good to ~0.01 deg."""
    n = jd - 2451545.0
    L = (280.460 + 0.9856474 * n) % 360.0
    g = math.radians((357.528 + 0.9856003 * n) % 360.0)
    lam = math.radians(L + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g))
    eps = math.radians(23.439 - 0.0000004 * n)
    ra = math.degrees(math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))) % 360.0
    dec = math.degrees(math.asin(math.sin(eps) * math.sin(lam)))
    return ra, dec


def twilight_phase(sun_alt: float) -> str:
    if sun_alt > 0:
        return "daylight"
    if sun_alt > -6:
        return "civil twilight"
    if sun_alt > -12:
        return "nautical twilight"
    if sun_alt > -18:
        return "astronomical twilight"
    return "night"


def altaz(ra_deg, dec_deg, R):
    a, d = math.radians(ra_deg), math.radians(dec_deg)
    e = np.array([math.cos(d) * math.cos(a), math.cos(d) * math.sin(a), math.sin(d)])
    x, y, z = R @ e
    alt = math.degrees(math.asin(max(-1, min(1, z))))
    az = math.degrees(math.atan2(x, y)) % 360.0  # from north, clockwise
    return alt, az


# --------------------------------------------------------------------------- #
# image helpers (Blender: rows bottom-up, RGBA float)
# --------------------------------------------------------------------------- #
def load_exr(path: str) -> np.ndarray:
    img = bpy.data.images.load(path, check_existing=True)
    w, h = img.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    arr = buf.reshape(h, w, 4)[:, :, :3]
    bpy.data.images.remove(img)
    return arr  # row 0 = bottom (south celestial pole)


def save_exr(path: str, rgb: np.ndarray) -> None:
    h, w, _ = rgb.shape
    img = bpy.data.images.new("out_sky", w, h, float_buffer=True, alpha=False)
    rgba = np.concatenate([rgb.astype(np.float32), np.ones((h, w, 1), np.float32)], axis=2)
    img.pixels.foreach_set(rgba.ravel())
    img.filepath_raw = path
    img.file_format = "OPEN_EXR"
    img.save()
    bpy.data.images.remove(img)


def save_preview(exr_path: str, png_path: str, width: int = 2048, exposure: float = 3.0) -> None:
    scene = bpy.context.scene
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Punchy"
    scene.view_settings.exposure = exposure
    img = bpy.data.images.load(exr_path)
    img.scale(width, width // 2)
    img.save_render(png_path)
    bpy.data.images.remove(img)


def bilinear(src: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Sample src (H,W,3) at continuous coords u,v in [0,1) (u wraps, v clamps)."""
    h, w, _ = src.shape
    x = u * w - 0.5
    y = v * h - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0).astype(np.float32)[..., None]
    fy = (y - y0).astype(np.float32)[..., None]
    x1 = (x0 + 1) % w
    x0 %= w
    y1 = np.clip(y0 + 1, 0, h - 1)
    y0 = np.clip(y0, 0, h - 1)
    out = (src[y0, x0] * (1 - fx) * (1 - fy) + src[y0, x1] * fx * (1 - fy)
           + src[y1, x0] * (1 - fx) * fy + src[y1, x1] * fx * fy)
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def fetch_starmap(res: str) -> str:
    os.makedirs(STARMAP_DIR, exist_ok=True)
    path = os.path.join(STARMAP_DIR, f"starmap_2020_{res}.exr")
    if not os.path.exists(path):
        url = SVS.format(res=res)
        print(f"[sky] downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path


def build(args) -> None:
    t = dt.datetime.fromisoformat(args.utc)
    jd = julian_date(t)
    lst = (gmst_deg(jd) + LON_DEG) % 360.0
    R = horizon_basis(LAT_DEG, lst)
    print(f"[sky] {args.utc} UTC  JD={jd:.5f}  GMST={gmst_deg(jd):.3f} deg  LST={lst:.3f} deg ({lst / 15:.3f} h)  lat={LAT_DEG} lon={LON_DEG}")
    sun_alt, sun_az = altaz(*sun_radec(jd), R)
    print(f"[sky] sun: alt {sun_alt:+.2f} deg  az {sun_az:.1f} deg  -> {twilight_phase(sun_alt)}")
    print(f"[sky] {'object':22s} {'alt':>7s} {'az':>7s}")
    for name, (ra, dec) in REFS.items():
        alt, az = altaz(ra, dec, R)
        print(f"[sky] {name:22s} {alt:7.1f} {az:7.1f}")

    src = load_exr(fetch_starmap(args.res))
    H, W, _ = src.shape
    lum = 0.2126 * src[..., 0] + 0.7152 * src[..., 1] + 0.0722 * src[..., 2]
    q = np.percentile(lum, [50, 90, 99, 99.9, 99.99])
    print(f"[sky] source {W}x{H}: luminance p50={q[0]:.4g} p90={q[1]:.4g} p99={q[2]:.4g} p99.9={q[3]:.4g} p99.99={q[4]:.4g} max={lum.max():.4g}")
    # normalise so the Milky Way band (p99 of the whole sky) sits at args.band
    scale = args.scale if args.scale else args.band / max(q[2], 1e-9)
    print(f"[sky] scale factor {scale:.4g}")

    # --- output grid in Blender's env convention --------------------------------
    # u: 0.5 -> +X (east), 0.25 -> +Y (north), 0.75 -> -Y, 0/1 -> -X; v: 0 bottom (nadir), 1 top (zenith)
    outW, outH = W, H
    u = (np.arange(outW, dtype=np.float64) + 0.5) / outW
    v = (np.arange(outH, dtype=np.float64) + 0.5) / outH
    phi = (0.5 - u) * 2 * math.pi          # atan2(y, x)
    el = (v - 0.5) * math.pi               # elevation
    cphi, sphi = np.cos(phi), np.sin(phi)
    cel, sel = np.cos(el), np.sin(el)
    dx = cel[:, None] * cphi[None, :]
    dy = cel[:, None] * sphi[None, :]
    dz = np.broadcast_to(sel[:, None], (outH, outW))
    # scene dir -> equatorial: e = R^T d
    ex = R[0, 0] * dx + R[1, 0] * dy + R[2, 0] * dz
    ey = R[0, 1] * dx + R[1, 1] * dy + R[2, 1] * dz
    ez = R[0, 2] * dx + R[1, 2] * dy + R[2, 2] * dz
    ra = np.degrees(np.arctan2(ey, ex)) % 360.0
    dec = np.degrees(np.arcsin(np.clip(ez, -1, 1)))
    # source map: centred on RA 0h, RA increasing to the LEFT; Dec +90 at the top row
    us = (0.5 - ra / 360.0) % 1.0
    vs = (dec + 90.0) / 180.0
    del ex, ey, ez, ra, dec
    print("[sky] resampling ...")
    out = bilinear(src, us, vs).astype(np.float32) * scale
    del us, vs, src

    # --- airglow + light pollution ---------------------------------------------------
    alt_deg = np.degrees(el)[:, None]
    above = (alt_deg > 0).astype(np.float32)
    az_deg = (np.degrees(np.arctan2(dx, dy)) % 360.0)  # compass azimuth per pixel (H,W)
    glow = np.zeros((outH, outW, 3), np.float32)
    airglow = args.band * 0.10 * np.exp(-np.clip(alt_deg, 0, 90) / 25.0) + args.band * 0.03
    glow += (airglow * above)[..., None] * np.array([0.85, 0.95, 1.0], np.float32)
    if not args.no_skyglow:
        for az_c, w_c, spread, hscale, col in CITIES:
            daz = (az_deg - az_c + 180.0) % 360.0 - 180.0
            g = w_c * np.exp(-0.5 * (daz / spread) ** 2) * np.exp(-np.clip(alt_deg, 0, 90) / hscale)
            glow += (g * above * args.band * args.skyglow)[..., None] * np.array(col, np.float32)
    out += glow
    # below the horizon: dark ground (never seen, terrain covers it) with a soft roll-off
    below = (alt_deg <= 0).astype(np.float32)
    out = out * (1 - below[..., None]) + below[..., None] * np.array([0.0015, 0.0015, 0.002], np.float32)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_exr(args.out, out)
    print(f"[sky] wrote {args.out} ({outW}x{outH})")

    # --- verification: predicted star positions must land on bright pixels ----------
    print(f"[sky] {'check':22s} {'alt':>6s} {'az':>6s} {'peak':>9s} {'median':>9s}")
    lum_o = 0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]
    for name, (ra_, dec_) in REFS.items():
        alt, az = altaz(ra_, dec_, R)
        if alt < 2 or "Sgr" in name:
            continue
        # pixel of predicted (az, alt)
        phi_p = math.radians(90.0 - az)
        up_ = (0.5 - phi_p / (2 * math.pi)) % 1.0
        vp = 0.5 + math.radians(alt) / math.pi
        cx, cy = int(up_ * outW), int(vp * outH)
        r = max(2, int(0.35 / 360 * outW))
        win = lum_o[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1]
        bg = lum_o[max(0, cy - 8 * r):cy + 8 * r + 1, max(0, cx - 8 * r):cx + 8 * r + 1]
        print(f"[sky] {name:22s} {alt:6.1f} {az:6.1f} {win.max():9.4g} {np.median(bg):9.4g}")

    # sidecar so the scene can put the twilight sun at the same instant as these stars
    gc_alt, gc_az = altaz(*REFS["Sgr A* (gal. centre)"], R)
    meta = {
        "utc": args.utc,
        "julian_date": jd,
        "lst_deg": lst,
        "latitude": LAT_DEG,
        "longitude": LON_DEG,
        "sun_altitude_deg": sun_alt,
        "sun_azimuth_deg": sun_az,
        "twilight_phase": twilight_phase(sun_alt),
        "galactic_centre_altitude_deg": gc_alt,
        "galactic_centre_azimuth_deg": gc_az,
        "source": "NASA/GSFC SVS Deep Star Maps 2020 (public domain); Gaia DR2: ESA/Gaia/DPAC",
        "starmap_resolution": args.res,
        "skyglow": 0.0 if args.no_skyglow else args.skyglow,
    }
    side = os.path.splitext(args.out)[0] + ".json"
    with open(side, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[sky] wrote {side}")

    if args.preview:
        save_preview(args.out, args.preview)
        print(f"[sky] wrote preview {args.preview}")


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--utc", default="2026-09-10T00:32", help="UTC instant (ISO). Default = 19:32 CDT, 9 Sep 2026: sun 4.4 deg below the horizon (civil twilight), galactic centre 19 deg up in the south")
    p.add_argument("--res", default="8k", choices=["4k", "8k", "16k"], help="NASA star-map resolution to use")
    p.add_argument("--out", default=os.path.join(HDRI_DIR, "fermilab_night_sky.exr"))
    p.add_argument("--preview", default=os.path.join(HDRI_DIR, "fermilab_night_sky_preview.png"))
    p.add_argument("--band", type=float, default=0.045, help="target radiance of the Milky Way band (p99 of sky)")
    p.add_argument("--scale", type=float, default=0.0, help="override the auto brightness scale")
    p.add_argument("--skyglow", type=float, default=1.0, help="light-pollution multiplier (0 = none)")
    p.add_argument("--no-skyglow", action="store_true")
    args = p.parse_args(argv)
    if bpy is None:
        sys.exit("run inside Blender: blender -b --python tools/make_sky_hdri.py -- ...")
    build(args)


if __name__ == "__main__":
    main()
