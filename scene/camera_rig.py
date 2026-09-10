"""Camera presets. Full-frame sensor, fast prime, depth of field on and
focused on Wilson Hall.

Note on DOF: at aerial distances (hundreds of metres) even f/1.4 gives a
physically tiny circle of confusion, so the effect is subtle by design;
the 'low' preset adds out-of-focus prairie grass a few metres in front of
the lens where a fast lens really shows.
"""
from __future__ import annotations

import math

import bpy
from mathutils import Vector

from . import common as C

PRESETS = {
    # name: dict(location, target, lens, fstop)

    # --- the hero vantage: NE of Wilson Hall looking SW, so the east-end profile sits over
    # the reflecting pond with the collider arc behind it and (on a September evening) the
    # galactic centre low in the SSW, directly above the building.
    "northeast": dict(location=(270.0, 400.0, 110.0), target=(-10.0, -20.0, 84.0), lens=35.0, fstop=1.8),
    # same vantage in cover format: tilted up so the building sits in the lower third and
    # the Milky Way fills the upper two-thirds. Use with a portrait --res.
    "cover": dict(location=(240.0, 358.0, 99.0), target=(-8.0, -18.0, 121.0), lens=30.0, fstop=1.8),
    # high NE overlook: the whole site laid out to the SW -- Wilson Hall left of centre and
    # the collider ring sweeping through the middle distance. Reads as a graphic, not a portrait.
    "overlook": dict(location=(760.0, 1020.0, 330.0), target=(500.0, -520.0, 40.0), lens=28.0, fstop=2.2),
    # --- the whole complex, north up. Due south of the site centre at 4.6 km and 2.6 km up,
    # looking due north on a 16 mm lens, so north on the ground projects exactly vertical.
    # Chosen by projecting the campus boundary over a grid of lens, distance, height and
    # shift: this fills the most frame width while keeping the whole site inside, the local
    # ground angle near 30 deg, and about a fifth of the frame as sky.
    # Needs a 3:2 frame -- 16:9 is too short to hold both sky and a 5 km-deep site, and gives
    # half the sky for the same camera. Render at e.g. 1600x1067 or 3840x2560.
    # NOTE: north up means looking away from the galactic centre, which is in the southern
    # sky, so the Milky Way core is behind the camera. Use "overview_south" to keep it.
    # Chosen by projecting the real boundary and the three largest rings through
    # a pinhole model of this camera and scoring the result, rather than by eye.
    # The subject used to occupy 11 % of the frame with 14 % of the height empty
    # dark ground below it, so the brightest, largest thing in the picture was
    # bare sky -- annotation cannot manufacture a hierarchy the tonal
    # composition denies.
    #
    # The model is validated against Blender's own projection dump and agrees
    # to 0.1 px on every anchor. It did not always: for several rounds its
    # shift_y sign was inverted, and because the scale and the horizontal axis
    # matched to four decimals the whole time, nothing looked wrong. What it
    # cost was the sky. At the framing that error produced, the true horizon sat
    # at y_img -0.10 -- above the top of the frame -- so ground filled the
    # picture and the bright specks along the top edge were distant street
    # lamps, not stars.
    #
    # Re-solved after the brief changed: zoom out so the whole campus reads and
    # the callouts cannot sit on the site outline, tilt up for more sky.
    #
    # The binding constraint is now that the *boundary polygon itself* must fall
    # between the two callout columns. Trimming the label copy is what made that
    # affordable: the widest text block is 168 px, so the columns need only
    # 0.115 of the width each and the drawing gets the middle 64 % rather than
    # 48 %. Solved at 18 mm from (1522, -5500, 2600): 23.8 deg depression, the
    # outline inside u 0.21..0.77, the site between y_img 0.43 and 0.75, 20 % of
    # the height sky, and the far edge at 0.40 of the near edge's scale.
    #
    # The site is deliberately smaller than it was -- 19 % of frame area against
    # 28 % -- because "zoom out so the text does not block it" and "make the
    # subject fill the frame" are opposed, and the brief chose the former.
    # 16 mm gives the same area at 0.36 uniformity, 14 mm at 0.35; 18 mm is the
    # widest that keeps the projection honest.
    "overview": dict(location=(1522.0, -5500.0, 2600.0), target=(1522.0, 387.0, 0.0),
                     lens=18.0, fstop=8.0, shift_y=0.020),
    # the earlier framing: looking south-south-west, galactic centre in frame, north down-right
    # The Milky Way view, and the one place the three-way conflict is resolved
    # rather than dodged. The galactic centre sits due south at 19.1 deg
    # altitude, so keeping it in frame bounds the depression angle by the lens:
    # at 24 mm the camera has to be within 4.4 deg of level, which foreshortens
    # the rings into slivers; only 16 mm and wider hold both a readable plan
    # angle and the core. Solved at 15 mm: 17.7 deg depression, the site across
    # 24 % of the frame, and the galactic centre at (0.52, 0.94) with 40 % of
    # frame height of sky between it and the site.
    #
    # The cost is that north now points away from the camera, so this view
    # cannot be north-up. That is a choice for the figure to make, not for the
    # camera: "overview" is the north-up site plan, this is the cover plate.
    "overview_south": dict(location=(1522.0, 5387.0, 1600.0), target=(1522.0, 387.0, 0.0),
                           lens=15.0, fstop=8.0, shift_y=-0.020),

    # --- other vantages. Note that from the SW the galactic centre is behind the camera,
    # so these see the fainter anti-centre sky.
    "aerial": dict(location=(-470.0, -430.0, 150.0), target=(20.0, 10.0, 100.0), lens=40.0, fstop=1.8),
    "aerial_wide": dict(location=(-560.0, -520.0, 190.0), target=(120.0, 40.0, 30.0), lens=32.0, fstop=2.0),
    "east": dict(location=(560.0, -330.0, 120.0), target=(0.0, 0.0, 90.0), lens=40.0, fstop=1.8),
    "high": dict(location=(-900.0, -1500.0, 520.0), target=(500.0, -300.0, 0.0), lens=35.0, fstop=2.8),
    "low": dict(location=(-330.0, -430.0, 42.0), target=(0.0, 0.0, 40.0), lens=32.0, fstop=1.4),
    "portrait": dict(location=(-420.0, -390.0, 140.0), target=(10.0, 5.0, 40.0), lens=35.0, fstop=1.8),
}


def make_camera(name, preset, focus_obj, *, col=None, aspect=(16, 9), lens=None, fstop=None, shift_x=None, shift_y=None):
    p = PRESETS[preset]
    cam = bpy.data.cameras.new(name)
    cam.sensor_width = 36.0
    cam.sensor_fit = "HORIZONTAL" if aspect[0] >= aspect[1] else "VERTICAL"
    cam.lens = lens or p["lens"]
    cam.clip_start = 0.5
    cam.clip_end = 80000.0
    cam.dof.use_dof = True
    cam.dof.focus_object = focus_obj
    cam.dof.aperture_fstop = fstop or p["fstop"]
    cam.dof.aperture_blades = 9
    cam.dof.aperture_rotation = math.radians(10)
    # A shift lens repositions the frame without changing the perspective, which
    # is how an architectural photographer fits a tall or deep subject: here it
    # lifts a 5 km-deep site clear of the footer band instead of tilting.
    cam.shift_x = p.get("shift_x", 0.0) if shift_x is None else shift_x
    cam.shift_y = p.get("shift_y", 0.0) if shift_y is None else shift_y
    obj = bpy.data.objects.new(name, cam)
    obj.location = p["location"]
    C.link_object(obj, col)
    C.aim(obj, p["target"])
    return obj


# Where a camera move starts, per preset. Solved with the same validated
# pinhole model as the framing itself, against two requirements: the whole site
# must already be inside the frame at frame 1, so nothing pops in during the
# move, and the change has to be felt without being a swoop.
#
# For "overview" the start is 1000 m lower and 600 m nearer, which puts the
# camera at 16.8 deg of depression against the final 23.8 deg and shows 30 % of
# the frame as sky against the final 20 %. The camera therefore rises and the
# horizon settles downward -- a move toward the sky that eases into the still.
# The lens and the shift do not change: this is a camera move, not a zoom, so
# the perspective the framing was solved for is the perspective it lands on.
APPROACH = {
    "overview": dict(location=(1522.0, -4900.0, 1600.0)),
    "overview_south": dict(location=(1522.0, 4600.0, 1050.0)),
}


def _action_fcurves(action):
    """F-curves of an action, across Blender's two Action APIs.

    Blender 4.4 replaced the flat `action.fcurves` with slotted actions --
    layers, strips and channelbags -- and 5.x drops the old attribute entirely,
    so reaching for `action.fcurves` raises AttributeError rather than
    returning nothing. Both shapes are handled, and an unrecognised one is
    reported instead of silently leaving the curves on their default
    interpolation, which would give a linear move with no easing at all.
    """
    if hasattr(action, "fcurves"):
        return list(action.fcurves)
    out = []
    for layer in getattr(action, "layers", []):
        for strip in getattr(layer, "strips", []):
            for bag in getattr(strip, "channelbags", []):
                out.extend(bag.fcurves)
    if not out:
        raise RuntimeError(
            "could not reach the camera action's f-curves on this Blender "
            f"({bpy.app.version_string}); the move would render without easing")
    return out


def animate_approach(cam_obj, scene, preset, *, seconds=15.0, fps=30):
    """Keyframe a camera move from its APPROACH pose into the preset pose.

    Two keyframes on a sine ease-in-out rather than a long path: the brief asks
    for subtle, and a two-pose ease is what reads as a slow settle rather than
    a fly-through. Rotation is keyframed alongside position, computed by aiming
    at the same target from each end, so the tilt change is exactly the
    consequence of the rise and nothing else moves.

    Returns (start_location, end_location, frames).
    """
    p = PRESETS[preset]
    start = APPROACH.get(preset, {}).get("location")
    if start is None:
        raise KeyError(f"no APPROACH pose for camera preset {preset!r}; add one "
                       "solved against the framing model rather than guessed")
    frames = max(int(round(seconds * fps)), 2)
    scene.frame_start = 1
    scene.frame_end = frames
    scene.render.fps = fps

    for frame, loc in ((1, start), (frames, p["location"])):
        cam_obj.location = loc
        C.aim(cam_obj, p["target"])
        cam_obj.keyframe_insert(data_path="location", frame=frame)
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=frame)

    for fc in _action_fcurves(cam_obj.animation_data.action):
        for kp in fc.keyframe_points:
            kp.interpolation = "SINE"
            kp.easing = "EASE_IN_OUT"
        fc.update()
    # leave the camera on its final pose so a still rendered from the same
    # scene is unaffected by the animation data
    cam_obj.location = p["location"]
    C.aim(cam_obj, p["target"])
    return start, p["location"], frames


def add_foreground_grass(cam_obj, col, *, count=70, seed=5):
    """Big-bluestem seed heads 2-5 m in front of a low camera: soft bokeh
    silhouettes across the bottom of the frame."""
    import random

    rng = random.Random(seed)
    mat = C.flat_material("fg_grass", (0.25, 0.20, 0.12), roughness=0.8)
    import bmesh

    inv = cam_obj.matrix_world
    for i in range(count):
        depth = rng.uniform(2.2, 5.5)
        x = rng.uniform(-0.55, 0.55) * depth
        y = rng.uniform(-0.62, -0.22) * depth
        base = inv @ Vector((x, y, -depth))
        bm = bmesh.new()
        h = rng.uniform(1.4, 2.4)
        w = 0.012
        lean = Vector((rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15), 0))
        verts = []
        for k in range(6):
            t = k / 5
            p = base + Vector((0, 0, h * t)) + lean * t * t * h
            verts.append((bm.verts.new(p + Vector((-w * (1 - 0.6 * t), 0, 0))), bm.verts.new(p + Vector((w * (1 - 0.6 * t), 0, 0)))))
        for k in range(5):
            bm.faces.new((verts[k][0], verts[k][1], verts[k + 1][1], verts[k + 1][0]))
        # seed head
        top = base + Vector((0, 0, h)) + lean * h
        bmesh.ops.create_icosphere(bm, subdivisions=1, radius=0.03, matrix=__import__("mathutils").Matrix.Translation(top))
        C.bmesh_object(f"fg_grass_{i}", bm, col=col, material=mat, recalc=False)
