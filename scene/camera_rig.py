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
    # composition denies. This framing puts the site across 76 % of the width
    # and 35 % of the frame area, leaves 5 % of dead foreground, and still keeps
    # a 14 % band of sky, which is the whole reason for a wide lens here.
    #
    # 24 mm is a deliberate midpoint: 20 mm gives a larger subject but projects
    # the far edge at 0.34 of the near edge's scale, and 50 mm is nearly uniform
    # at 0.61 but shrinks the site to 24 % of frame. At 24 mm the ratio is 0.41.
    # The target is the site-filler centre, so the chain sits in the middle of
    # the campus; the old target predated the measured plan by about 1.5 km.
    "overview": dict(location=(1522.0, -5500.0, 2800.0), target=(1522.0, 387.0, 0.0),
                     lens=24.0, fstop=8.0, shift_y=-0.080),
    # the earlier framing: looking south-south-west, galactic centre in frame, north down-right
    "overview_south": dict(location=(3277.0, 4538.0, 4359.0), target=(1225.0, -1100.0, 0.0),
                           lens=24.0, fstop=8.0, shift_x=0.030, shift_y=-0.027),

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
