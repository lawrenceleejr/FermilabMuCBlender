"""Render every camera listed in the scene's `render_cameras` property.

    blender --background out/<style>.blend --python scripts/render_stills.py -- \
        --outdir out/renders
"""

import argparse
import os
import sys

import bpy


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/renders")
    args = ap.parse_args(argv)

    sc = bpy.context.scene
    style = sc.get("style", "scene")
    cams = list(sc.get("render_cameras", []))
    if not cams:
        cams = [o.name for o in sc.objects if o.type == "CAMERA"]
    os.makedirs(args.outdir, exist_ok=True)

    # High-quality JPEG stills keep the uploaded artifact under the 32 MB
    # limit (a 1440p PNG is ~4 MB; the same frame as JPEG q92 is ~0.7 MB).
    sc.render.image_settings.file_format = "JPEG"
    sc.render.image_settings.quality = 92
    ext = "jpg"

    annotations = bpy.data.collections.get("Annotations")
    for name in cams:
        cam = bpy.data.objects.get(name)
        if cam is None:
            print(f"WARNING: camera {name!r} missing, skipping")
            continue
        sc.camera = cam
        if annotations is not None:
            # the legend/title live in world space near the ground; hide them
            # for the ground-level hero shot where they'd float on the horizon
            annotations.hide_render = (name == "Cam_WilsonHall")
        path = os.path.abspath(os.path.join(args.outdir, f"{style}_{name}.{ext}"))
        sc.render.filepath = path
        print(f"rendering {name} -> {path}")
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
