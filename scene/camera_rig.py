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
    "aerial": dict(location=(-470.0, -430.0, 150.0), target=(20.0, 10.0, 100.0), lens=40.0, fstop=1.8),
    "east": dict(location=(560.0, -330.0, 120.0), target=(0.0, 0.0, 90.0), lens=40.0, fstop=1.8),
    # NE of Wilson Hall looking SW: the east-end profile over the reflecting pond, the collider
    # arc behind, and (Sept evening) the galactic centre low in the SSW above the building
    "northeast": dict(location=(330.0, 470.0, 125.0), target=(-20.0, -30.0, 95.0), lens=32.0, fstop=1.8),
    "aerial_wide": dict(location=(-560.0, -520.0, 190.0), target=(120.0, 40.0, 30.0), lens=32.0, fstop=2.0),
    "high": dict(location=(-900.0, -1500.0, 520.0), target=(500.0, -300.0, 0.0), lens=35.0, fstop=2.8),
    "low": dict(location=(-330.0, -430.0, 42.0), target=(0.0, 0.0, 40.0), lens=32.0, fstop=1.4),
    "portrait": dict(location=(-420.0, -390.0, 140.0), target=(10.0, 5.0, 40.0), lens=35.0, fstop=1.8),
}


def make_camera(name, preset, focus_obj, *, col=None, aspect=(16, 9), lens=None, fstop=None, shift_y=0.0):
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
    cam.shift_y = shift_y
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
