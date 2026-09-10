"""Build and render the Fermilab muon-collider hero image.

Run with Blender >= 5:
    blender -b -P scene/build_scene.py -- [options]

Options (after the `--`):
  --out PATH             output PNG (default renders/fermilab_muc_cover.png)
  --res WxH              resolution (default 2560x1440)
  --samples N            Cycles samples (default 512, adaptive)
  --adaptive-threshold T Cycles noise threshold, the stop condition (default 0.012;
                         0.01 for finals via --final)
  --time-limit SEC       stop sampling after N seconds per image (0 = off)
  --camera NAME          northeast (default) | overview (whole complex) | cover (portrait)
                         | overlook | aerial | east | aerial_wide | high | low | portrait
  --lens MM --fstop F    override the preset lens / aperture
  --sky twilight|milkyway|nishita|hdri
                         twilight (default) = physical twilight sky + the real star field;
                         milkyway = star field alone (full night); nishita = sky alone;
                         hdri = a Poly Haven sky. Sun position comes from the star map's
                         JSON sidecar unless --sun-elevation/--sun-azimuth override it.
  --star-scale S         multiplier on the star field in twilight mode (default 1.0)
  --boundary auto|on|off highlight the site boundary; auto (default) = only the wide cameras
  --boundary-strength S  how brightly the boundary reads to the camera (default 4.0)
  --towns                add distant town-glow strips (off: the sky HDRI already has sky glow)
  --sky-file PATH        pre-oriented night-sky EXR (default assets/hdri/fermilab_night_sky.exr)
  --device CPU|GPU       Cycles device (default CPU; use GPU on a workstation)
  --moon                 add a moon (off by default: it would wash out the Milky Way)
  --sun-elevation DEG    override sun elevation (default: from the sky sidecar)
  --sun-azimuth DEG      override sun compass azimuth (default: from the sky sidecar)
  --hdri NAME            Poly Haven id (default kloppenheim_06_puresky)
  --hdri-res 2k|4k       which downloaded resolution to use (default 4k, falls back)
  --sky-strength S       sky multiplier (default 1.3 twilight / 1.0 nishita / 0.12 hdri)
  --sky-rot DEG          rotate the sky about Z (default 161.6: sunset glow at WNW)
  --exposure EV          view exposure (default 2.0 twilight / 0.95 milkyway)
  --fog-density D        ground fog peak density per metre (default 3.2e-3)
  --no-fog               disable the ground-fog volume
  --no-haze              disable the aerial haze volume (--haze-density D to tune)
  --tree-density F       scale tree counts (default 1.0; 0.3 for quick previews)
  --no-grass             skip foreground grass on the low camera
  --annotations PATH     write JSON projection of named features (for tools/annotate.py)
  --save-blend PATH      also save the .blend
  --no-render            build/save only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import bpy
import mathutils

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from scene import anim, atmosphere, camera_rig, common as C, postfx, site, wilson_hall  # noqa: E402

# Cameras high/wide enough that the site-boundary ribbon reads as an outline on the ground
# rather than a line across the horizon.
# The site outline reads from any vantage high enough to see the whole campus.
# overview_south was omitted here by oversight -- it is the cover framing and
# shows the site at the same scale as "overview", just from the north.
BOUNDARY_CAMERAS = {"overview", "overview_south", "overlook", "high"}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(C.ROOT, "renders", "fermilab_muc_cover.png"))
    p.add_argument("--res", default="2560x1440")
    p.add_argument("--samples", type=int, default=512)
    p.add_argument("--time-limit", type=float, default=0.0)
    p.add_argument("--adaptive-threshold", type=float, default=0.012, help="Cycles adaptive-sampling noise threshold; this is what ends the render, with --samples as the ceiling. Lower = cleaner and slower (0.01 for finals)")
    p.add_argument("--camera", default="northeast", choices=sorted(camera_rig.PRESETS))
    p.add_argument("--lens", type=float, default=None)
    p.add_argument("--fstop", type=float, default=None)
    p.add_argument("--sky", default="twilight", choices=["twilight", "milkyway", "nishita", "hdri"])
    p.add_argument("--sky-file", default=os.path.join(C.HDRI_DIR, "fermilab_night_sky.exr"), help="pre-oriented night-sky EXR from tools/make_sky_hdri.py (milkyway mode)")
    p.add_argument("--device", default="CPU", choices=["CPU", "GPU"], help="Cycles compute device (GPU auto-selects OPTIX/CUDA/HIP/METAL/ONEAPI)")
    p.add_argument("--moon", action="store_true", help="add a moon (off by default: it would wash out the Milky Way)")
    p.add_argument("--sun-elevation", type=float, default=None, help="sun elevation in degrees (negative = below horizon); default comes from the sky sidecar")
    p.add_argument("--sun-azimuth", type=float, default=None, help="sun compass azimuth (deg from north, clockwise); default comes from the sky sidecar")
    p.add_argument("--star-scale", type=float, default=1.0, help="multiplier on the star field in twilight mode")
    p.add_argument("--boundary", default="auto", choices=["auto", "on", "off"],
                   help="highlight the Fermilab site boundary: auto (default) enables it only for the wide cameras, where it explains the site's extent instead of streaking across the horizon")
    p.add_argument("--no-boundary", action="store_true", help="alias for --boundary off")
    p.add_argument("--boundary-strength", type=float, default=4.0, help="how brightly the site boundary reads to the camera")
    p.add_argument("--hdri", default="kloppenheim_06_puresky")
    p.add_argument("--hdri-res", default="4k")
    p.add_argument("--sky-strength", type=float, default=None, help="sky multiplier (default 1.3 for twilight, 1.0 for nishita, 0.12 for hdri)")
    p.add_argument("--sky-rot", type=float, default=161.6)
    p.add_argument("--exposure", type=float, default=None, help="view exposure in stops (default 2.0 for twilight, 0.95 for milkyway)")
    p.add_argument("--fog-density", type=float, default=3.2e-3)
    p.add_argument("--no-fog", action="store_true")
    p.add_argument("--no-haze", action="store_true")
    p.add_argument("--haze-density", type=float, default=3.0e-5)
    p.add_argument("--no-stars", action="store_true")
    p.add_argument("--tree-density", type=float, default=1.0)
    p.add_argument("--no-grass", action="store_true")
    p.add_argument("--no-floods", action="store_true")
    p.add_argument("--towns", action="store_true", help="add distant town-glow strips (the sky HDRI already carries light-pollution domes; from an elevated camera the strips read as a dark bar against the ground)")
    p.add_argument("--moon-az", type=float, default=62.0)
    p.add_argument("--moon-el", type=float, default=9.0)
    p.add_argument("--moon-energy", type=float, default=0.12)
    p.add_argument("--annotations", default="", help="write a JSON projection of the named site features for tools/annotate.py")
    p.add_argument("--save-blend", default="")
    p.add_argument("--animate", type=float, default=0.0,
                   help="render a camera-approach movie of this many seconds instead of a still")
    p.add_argument("--fps", type=int, default=30, help="frame rate for --animate")
    p.add_argument("--no-descent", action="store_true",
                   help="with --animate, keep the machines in the ground instead of lowering them in")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--threads", type=int, default=0)
    return p.parse_args(argv)


def sky_metadata(sky_path):
    """Load the JSON sidecar written next to the star map by tools/make_sky_hdri.py."""
    side = os.path.splitext(sky_path)[0] + ".json"
    if not os.path.exists(side):
        return {}
    try:
        with open(side) as fh:
            return json.load(fh)
    except (OSError, ValueError) as e:
        print(f"[build] could not read {side}: {e}")
        return {}


def hdri_path(name, res):
    for r in (res, "4k", "2k", "1k"):
        p = os.path.join(C.HDRI_DIR, f"{name}_{r}.hdr")
        if os.path.exists(p):
            return p
    return os.path.join(C.HDRI_DIR, f"{name}_{res}.hdr")


def dump_annotations(path, scene, cam, width, height, sky_meta):
    """Project the named site features through the render camera and write JSON.

    This is what lets tools/annotate.py put labels exactly on their subjects at
    any output resolution: the anchors are the same constants the geometry is
    built from, and the projection is the render camera's own.

    Coordinates are normalised (0..1, origin top-left) so they scale to any
    resolution. `depth_m` is distance along the camera axis, used to sort labels
    front-to-back. NOTE: the compositor's lens distortion (~0.4 %) is not
    modelled here -- it displaces a point by well under a pixel near the centre
    and at most ~3 px at the extreme corners of a 4K frame.
    """
    from bpy_extras.object_utils import world_to_camera_view

    # The camera was just built and aimed via a constraint-free matrix write, so
    # its evaluated matrix_world is stale until the depsgraph is flushed. Without
    # this every projection collapses to the origin.
    bpy.context.view_layer.update()

    anchors = site.annotation_anchors()

    def project(co):
        ndc = world_to_camera_view(scene, cam, mathutils.Vector(co))
        return ndc.x, 1.0 - ndc.y, ndc.z

    def resolve(a):
        """Fixed point, or the sample on a ring/path nearest the screen target."""
        if "pos" in a:
            return tuple(a["pos"])
        if "ring" in a:
            r = a["ring"]
            cx, cy = r["center"]
            rx, ry, z = r["radius"], r.get("ry") or r["radius"], r.get("z", 0.0)
            samples = [(cx + rx * math.cos(math.radians(t)), cy + ry * math.sin(math.radians(t)), z)
                       for t in range(0, 360, 2)]
        else:
            pts, z = a["path"], a.get("z", 0.0)
            samples = []
            for (x0, y0), (x1, y1) in zip(pts, list(pts[1:]) + [pts[0]]):
                for f in [i / 12.0 for i in range(12)]:
                    samples.append((x0 + (x1 - x0) * f, y0 + (y1 - y0) * f, z))
        tx, ty = a.get("prefer", (0.5, 0.5))
        best, best_d = None, 1e9
        for co in samples:
            px, py, depth = project(co)
            if depth <= 0 or not (0.02 <= px <= 0.98 and 0.02 <= py <= 0.98):
                continue
            d = (px - tx) ** 2 + (py - ty) ** 2
            if d < best_d:
                best, best_d = co, d
        return best if best is not None else tuple(samples[0])

    out = {
        "render": {"width": width, "height": height, "camera": cam.name},
        "camera": {
            "location": list(cam.location),
            "lens_mm": cam.data.lens,
            "sensor_mm": cam.data.sensor_width,
        },
        "sky": sky_meta,
        "features": {},
    }
    for key, a in anchors.items():
        co = resolve(a)
        px, py, depth = project(co)
        out["features"][key] = {
            "label": a["label"],
            "metric": a.get("metric", ""),
            "accent": a.get("accent", "neutral"),
            "world": list(co),
            # normalised, origin top-left, so the layout is resolution-independent
            "x": px,
            "y": py,
            "depth_m": depth,
            "on_screen": bool(0.0 <= px <= 1.0 and 0.0 <= py <= 1.0 and depth > 0),
        }

    # --- scale: pixels per metre on the ground at the site centroid --------------
    # A perspective view has no single scale, so the scale bar is quoted at the
    # centroid and labelled as such.
    centroid = mathutils.Vector((1225.0, -1100.0, 0.0))
    east = mathutils.Vector((1.0, 0.0, 0.0))
    p0 = world_to_camera_view(scene, cam, centroid)
    p1 = world_to_camera_view(scene, cam, centroid + east * 1000.0)
    dx = (p1.x - p0.x) * width
    dy = (p1.y - p0.y) * height
    # Directions are stored as pixel deltas in image coordinates (x right, y
    # DOWN) for 1 km on the ground, so the consumer never has to guess a sign.
    out["scale"] = {
        "reference_world": list(centroid),
        "reference_x": p0.x,
        "reference_y": 1.0 - p0.y,
        "px_per_km_at_reference": math.hypot(dx, dy),
        "east_px_per_km": [dx, -dy],
    }

    # --- north, projected at the same reference point -----------------------------
    n1 = world_to_camera_view(scene, cam, centroid + mathutils.Vector((0.0, 1000.0, 0.0)))
    ndx = (n1.x - p0.x) * width
    ndy = (n1.y - p0.y) * height
    out["north"] = {"north_px_per_km": [ndx, -ndy], "px_per_km": math.hypot(ndx, ndy)}

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    vis = sum(1 for f in out["features"].values() if f["on_screen"])
    print(f"[build] wrote {path}: {vis}/{len(out['features'])} features on screen, "
          f"{out['scale']['px_per_km_at_reference']:.1f} px/km at centroid, "
          f"north {math.degrees(math.atan2(-out['north']['north_px_per_km'][1], out['north']['north_px_per_km'][0])):.1f} deg")
    return out


def main():
    args = parse_args()
    if args.sky_strength is None:
        args.sky_strength = {"twilight": 1.3, "milkyway": 1.0, "nishita": 1.0, "hdri": 0.12}[args.sky]
    if args.exposure is None:
        # twilight is orders of magnitude brighter than night, so it needs far less lift
        args.exposure = {"twilight": 2.0, "nishita": 1.6, "milkyway": 0.95, "hdri": 0.6}[args.sky]
    t0 = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"

    cols = {k: C.new_collection(k) for k in ("site", "wilson_hall", "collider", "water", "roads", "lights", "buildings", "trees", "atmosphere", "cameras")}

    # --- materials shared across the site ----------------------------------
    prairie = site.prairie_material()
    water = C.water_material("lake_water")
    asphalt = site.asphalt_material()
    concrete = C.pbr_material("site_concrete", "Concrete042A_1K-JPG", scale=3.0, tint=(0.7, 0.68, 0.64), normal_strength=0.6)

    # --- geometry -------------------------------------------------------------
    site.build_terrain(cols["site"], prairie)
    focus = wilson_hall.build(cols["wilson_hall"], floodlights=not args.no_floods)
    site.build_rings(cols["site"], prairie, water)
    site.build_collider(cols["collider"], concrete)
    site.build_water(cols["water"], water)
    site.build_roads(cols["roads"], asphalt)
    site.build_street_lights(cols["lights"])
    site.build_buildings(cols["buildings"], concrete)
    site.build_village(cols["buildings"])
    ntrees = site.build_trees(cols["trees"], density=args.tree_density)
    # Seen edge-on from a low camera the boundary ribbon reads as a bright streak across the
    # horizon, so by default it is drawn only for the cameras high enough to look down on it.
    want_boundary = {"on": True, "off": False}.get(args.boundary, args.camera in BOUNDARY_CAMERAS)
    if args.no_boundary:
        want_boundary = False
    if want_boundary:
        site.build_campus_boundary(cols["site"], camera_strength=args.boundary_strength)
        print(f"[build] site boundary highlighted ({args.camera})")
    if args.towns:
        site.build_towns(cols["site"])
    print(f"[build] geometry done: {len(bpy.data.objects)} objects, {ntrees} trees, {time.time() - t0:.1f}s")

    # --- atmosphere -------------------------------------------------------------
    sky_path = args.sky_file if args.sky in ("twilight", "milkyway") else hdri_path(args.hdri, args.hdri_res)
    meta = sky_metadata(sky_path)
    sun_el = args.sun_elevation if args.sun_elevation is not None else meta.get("sun_altitude_deg", -6.8)
    sun_az = args.sun_azimuth if args.sun_azimuth is not None else meta.get("sun_azimuth_deg", 283.0)
    if meta:
        print(f"[build] sky {meta.get('utc')} UTC ({meta.get('twilight_phase')}): sun {sun_el:+.2f} deg alt / {sun_az:.1f} deg az, "
              f"galactic centre {meta.get('galactic_centre_altitude_deg', float('nan')):.1f} deg alt / {meta.get('galactic_centre_azimuth_deg', float('nan')):.1f} deg az")
    if args.sky in ("twilight", "milkyway") and not os.path.exists(sky_path):
        print(f"[build] WARNING: {sky_path} missing -- run: blender -b --python tools/make_sky_hdri.py -- --res 8k")
    atmosphere.build_world(sky_path, mode=args.sky, strength=args.sky_strength, rotation_deg=args.sky_rot,
                           sun_elevation_deg=sun_el, sun_azimuth_deg=sun_az, star_scale=args.star_scale,
                           stars=not args.no_stars and args.sky in ("nishita", "hdri"))
    if not args.no_haze:
        atmosphere.build_haze(cols["atmosphere"], density=args.haze_density)
    if not args.no_fog:
        atmosphere.build_ground_fog(cols["atmosphere"], density=args.fog_density)
    if args.moon:
        atmosphere.add_moon(cols["atmosphere"], azimuth_deg=args.moon_az, elevation_deg=args.moon_el, energy=args.moon_energy)

    # --- camera -----------------------------------------------------------------
    w, h = (int(v) for v in args.res.lower().split("x"))
    cam = camera_rig.make_camera("camera", args.camera, focus, col=cols["cameras"], aspect=(w, h), lens=args.lens, fstop=args.fstop)
    scene.camera = cam
    if args.camera == "low" and not args.no_grass:
        camera_rig.add_foreground_grass(cam, cols["cameras"])

    # --- render setup -------------------------------------------------------------
    postfx.configure_cycles(scene, samples=args.samples, adaptive_threshold=args.adaptive_threshold, time_limit=args.time_limit, threads=args.threads, device=args.device)
    postfx.configure_output(scene, width=w, height=h, path=os.path.abspath(args.out), exposure=args.exposure)
    postfx.build_compositor(scene)

    if args.annotations:
        dump_annotations(args.annotations, scene, cam, w, h, meta)

    if args.save_blend:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_blend)) or ".", exist_ok=True)
        blend_path = os.path.abspath(args.save_blend)
        bpy.ops.wm.save_as_mainfile(filepath=blend_path)
        try:  # make texture/HDRI paths relative so the file travels with the repo
            bpy.ops.file.make_paths_relative()
            bpy.ops.wm.save_as_mainfile(filepath=blend_path)
        except Exception as e:  # noqa: BLE001
            print("[build] relative paths:", e)
        print(f"[build] saved {blend_path}")
    print(f"[build] scene ready in {time.time() - t0:.1f}s")

    if args.animate:
        # A frame sequence, not a video file. Cycles writing straight to a
        # container gives no way to resume a run that dies at frame 300 of 450,
        # and a GPU pass long enough to matter is exactly the kind that dies.
        # ffmpeg turns the sequence into an mp4 in seconds afterwards.
        start, end, frames = camera_rig.animate_approach(
            cam, scene, args.camera, seconds=args.animate, fps=args.fps)
        # Motion blur off for the move. It is enabled as house style because a
        # still has nothing moving in it, so it costs nothing there. Here it
        # costs BVH work on 50 000 objects across every frame and buys nothing:
        # the approach covers 1000 m of rise and 7 deg of tilt over 450 frames,
        # so a 0.5 shutter smears about 0.07 px. (It is very visible if the same
        # move is compressed into a two-frame test, which is a property of the
        # test rather than of the movie.)
        scene.render.use_motion_blur = False
        if not args.no_descent:
            for label, n, z0, s, e in anim.animate_descent(scene, frames):
                print(f"[render]   {label}: {n} objects from {z0:.0f} m, "
                      f"frames {s}-{e} ({(e - s) / args.fps:.1f}s, "
                      f"{z0 / max((e - s) / args.fps, 1e-6):.0f} m/s)")
        stem = os.path.splitext(os.path.abspath(args.out))[0]
        os.makedirs(stem, exist_ok=True)
        scene.render.filepath = os.path.join(stem, "frame_")
        scene.render.image_settings.file_format = "PNG"
        print(f"[render] animating {args.animate:.0f}s at {args.fps} fps "
              f"= {frames} frames, {w}x{h}, {args.samples} spp")
        print(f"[render]   from {tuple(round(v) for v in start)} "
              f"to {tuple(round(v) for v in end)}")
        print(f"[render]   frames -> {scene.render.filepath}####.png")
        if not args.no_render:
            t1 = time.time()
            bpy.ops.render.render(animation=True)
            dt = time.time() - t1
            print(f"[render] {frames} frames in {dt / 60:.0f} min "
                  f"({dt / max(frames, 1):.1f}s per frame)")
            print(f"[render] encode with: ffmpeg -y -framerate {args.fps} "
                  f"-i {scene.render.filepath}%04d.png -c:v libx264 -pix_fmt yuv420p "
                  f"-crf 17 {stem}.mp4")
    elif not args.no_render:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        t1 = time.time()
        bpy.ops.render.render(write_still=True)
        print(f"[render] wrote {args.out} in {time.time() - t1:.0f}s ({w}x{h}, {args.samples} spp)")


if __name__ == "__main__":
    main()
