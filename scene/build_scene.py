"""Build and render the Fermilab muon-collider hero image.

Run with Blender >= 5:
    blender -b -P scene/build_scene.py -- [options]

Options (after the `--`):
  --out PATH             output PNG (default renders/fermilab_muc_cover.png)
  --res WxH              resolution (default 2560x1440)
  --samples N            Cycles samples (default 512, adaptive)
  --time-limit SEC       stop sampling after N seconds per image (0 = off)
  --camera NAME          northeast (default) | aerial | east | aerial_wide | high | low | portrait
  --lens MM --fstop F    override the preset lens / aperture
  --sky milkyway|nishita|hdri  real night sky from tools/make_sky_hdri.py (default), physically based twilight, or a Poly Haven HDRI
  --sky-file PATH        pre-oriented night-sky EXR (default assets/hdri/fermilab_night_sky.exr)
  --device CPU|GPU       Cycles device (default CPU; use GPU on a workstation)
  --moon                 add a moon (off by default: it would wash out the Milky Way)
  --sun-elevation DEG    Nishita sun elevation (default -4: civil twilight)
  --sun-azimuth DEG      Nishita sunset compass azimuth (default 290 = WNW)
  --hdri NAME            Poly Haven id (default kloppenheim_06_puresky)
  --hdri-res 2k|4k       which downloaded resolution to use (default 4k, falls back)
  --sky-strength S       sky multiplier (default 1.0 nishita / 0.12 hdri)
  --sky-rot DEG          rotate the sky about Z (default 161.6: sunset glow at WNW)
  --exposure EV          view exposure (default 0.6)
  --fog-density D        ground fog peak density per metre (default 2.0e-3)
  --no-fog               disable the ground-fog volume
  --no-haze              disable the aerial haze volume (--haze-density D to tune)
  --tree-density F       scale tree counts (default 1.0; 0.3 for quick previews)
  --no-grass             skip foreground grass on the low camera
  --save-blend PATH      also save the .blend
  --no-render            build/save only
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from scene import atmosphere, camera_rig, common as C, postfx, site, wilson_hall  # noqa: E402


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(C.ROOT, "renders", "fermilab_muc_cover.png"))
    p.add_argument("--res", default="2560x1440")
    p.add_argument("--samples", type=int, default=512)
    p.add_argument("--time-limit", type=float, default=0.0)
    p.add_argument("--camera", default="northeast", choices=sorted(camera_rig.PRESETS))
    p.add_argument("--lens", type=float, default=None)
    p.add_argument("--fstop", type=float, default=None)
    p.add_argument("--sky", default="milkyway", choices=["milkyway", "nishita", "hdri"])
    p.add_argument("--sky-file", default=os.path.join(C.HDRI_DIR, "fermilab_night_sky.exr"), help="pre-oriented night-sky EXR from tools/make_sky_hdri.py (milkyway mode)")
    p.add_argument("--device", default="CPU", choices=["CPU", "GPU"], help="Cycles compute device (GPU auto-selects OPTIX/CUDA/HIP/METAL/ONEAPI)")
    p.add_argument("--moon", action="store_true", help="add a moon (off by default: it would wash out the Milky Way)")
    p.add_argument("--sun-elevation", type=float, default=-4.0, help="Nishita sun elevation in degrees (negative = below horizon)")
    p.add_argument("--sun-azimuth", type=float, default=290.0, help="Nishita sunset compass azimuth (deg from north, clockwise)")
    p.add_argument("--hdri", default="kloppenheim_06_puresky")
    p.add_argument("--hdri-res", default="4k")
    p.add_argument("--sky-strength", type=float, default=None, help="sky multiplier (default 1.0 for nishita, 0.12 for hdri)")
    p.add_argument("--sky-rot", type=float, default=161.6)
    p.add_argument("--exposure", type=float, default=0.6)
    p.add_argument("--fog-density", type=float, default=2.0e-3)
    p.add_argument("--no-fog", action="store_true")
    p.add_argument("--no-haze", action="store_true")
    p.add_argument("--haze-density", type=float, default=3.5e-5)
    p.add_argument("--no-stars", action="store_true")
    p.add_argument("--tree-density", type=float, default=1.0)
    p.add_argument("--no-grass", action="store_true")
    p.add_argument("--no-floods", action="store_true")
    p.add_argument("--moon-az", type=float, default=62.0)
    p.add_argument("--moon-el", type=float, default=9.0)
    p.add_argument("--moon-energy", type=float, default=0.12)
    p.add_argument("--save-blend", default="")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--threads", type=int, default=0)
    return p.parse_args(argv)


def hdri_path(name, res):
    for r in (res, "4k", "2k", "1k"):
        p = os.path.join(C.HDRI_DIR, f"{name}_{r}.hdr")
        if os.path.exists(p):
            return p
    return os.path.join(C.HDRI_DIR, f"{name}_{res}.hdr")


def main():
    args = parse_args()
    if args.sky_strength is None:
        args.sky_strength = {"milkyway": 1.0, "nishita": 1.0, "hdri": 0.12}[args.sky]
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
    ntrees = site.build_trees(cols["trees"], density=args.tree_density)
    site.build_towns(cols["site"])
    print(f"[build] geometry done: {len(bpy.data.objects)} objects, {ntrees} trees, {time.time() - t0:.1f}s")

    # --- atmosphere -------------------------------------------------------------
    sky_path = args.sky_file if args.sky == "milkyway" else hdri_path(args.hdri, args.hdri_res)
    if args.sky == "milkyway" and not os.path.exists(sky_path):
        print(f"[build] WARNING: {sky_path} missing -- run: blender -b --python tools/make_sky_hdri.py; falling back to nishita")
        args.sky = "nishita"
    atmosphere.build_world(sky_path, mode=args.sky, strength=args.sky_strength, rotation_deg=0.0 if args.sky == "milkyway" else args.sky_rot, sun_elevation_deg=args.sun_elevation, sun_azimuth_deg=args.sun_azimuth, stars=not args.no_stars)
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
    postfx.configure_cycles(scene, samples=args.samples, time_limit=args.time_limit, threads=args.threads, device=args.device)
    postfx.configure_output(scene, width=w, height=h, path=os.path.abspath(args.out), exposure=args.exposure)
    postfx.build_compositor(scene)

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

    if not args.no_render:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        t1 = time.time()
        bpy.ops.render.render(write_still=True)
        print(f"[render] wrote {args.out} in {time.time() - t1:.0f}s ({w}x{h}, {args.samples} spp)")


if __name__ == "__main__":
    main()
