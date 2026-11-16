"""The flight: from the metro view (Fermilab and Chicago from 28 km) down to the
site overview, in one continuous camera move.

Two representations of the same ground share the scene and hand over during
the descent:

* the **metro layer** -- the curved, imagery-mapped ground with its lamp
  raster (scene/metro.py), the wide outlines, and the curved aerial haze;
* the **site layer** -- everything the site views are built from: prairie
  terrain, trees, the three registers of street lamps, Wilson Hall, the
  machines, the boundary ribbon, the flat aerial haze and the ground fog.

One animated scene property, `site_mix`, carries the handover. Every site
material is wrapped as mix(Transparent, material, site_mix), the metro
overlays and haze as the inverse, so the hand-over is a cross-dissolve in the
shaders, not a visibility pop -- and with site_mix at 1 the last frame is the
site overview exactly, the same camera on the same scene. While a layer is
fully faded out, its collections are also hidden from render, which keeps
the 50 000 site objects out of the early frames entirely.

The move itself is a dolly on a logarithmic distance scale: the camera's
distance to its aim point falls from 65 km to 6.4 km at a constant *ratio* per
frame, which is what reads as a steady approach (a linear distance would rush
the wide end and crawl at the close one). The aim point slides from between
the site and the Loop onto the site in proportion to the distance still to
cover, so it moves at a constant fraction of the frame rather than whipping
across in the last seconds. The look direction is slerped between the two
poses -- they differ by under 2 deg of pitch -- and the lens goes 26 -> 18 mm.
Time is eased with smootherstep, so the move starts and lands at rest.

The crossfade runs on altitude: metro above 9 km, site below 4.5 km. That is
where a pixel of the metro raster (15 m) stops being finer than a pixel of the
frame, and where the site layer's own detail -- trees, poles, the 11 m boundary
ribbon -- starts to resolve.
"""
from __future__ import annotations

import math

import bpy
from mathutils import Quaternion, Vector

from . import camera_rig
from . import metro_geo as G

PROP = "site_mix"


def _smoother(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * t * (t * (6 * t - 15) + 10)


def _smoothstep(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3 - 2 * t)


# --------------------------------------------------------------------------- #
# fades
# --------------------------------------------------------------------------- #
def _fac(nt, invert):
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_type = "VIEW_LAYER"
    attr.attribute_name = PROP
    if not invert:
        return attr.outputs["Fac"]
    m = nt.nodes.new("ShaderNodeMath")
    m.operation = "SUBTRACT"
    m.inputs[0].default_value = 1.0
    nt.links.new(attr.outputs["Fac"], m.inputs[1])
    return m.outputs[0]


def fade_material(mat, *, invert=False):
    """Wrap a material's surface as mix(transparent, surface, site_mix) and its
    volume as mix(nothing, volume, site_mix); `invert` fades the other way."""
    if not mat or not mat.use_nodes or mat.get("_flight_faded"):
        return
    nt = mat.node_tree
    for out in [n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"]:
        for sock_name in ("Surface", "Volume"):
            sock = out.inputs[sock_name]
            if not sock.links:
                continue
            src = sock.links[0].from_socket
            mix = nt.nodes.new("ShaderNodeMixShader")
            nt.links.new(_fac(nt, invert), mix.inputs["Fac"])
            if sock_name == "Surface":
                tr = nt.nodes.new("ShaderNodeBsdfTransparent")
                nt.links.new(tr.outputs["BSDF"], mix.inputs[1])
            nt.links.new(src, mix.inputs[2])
            nt.links.new(mix.outputs["Shader"], sock)
    mat["_flight_faded"] = True


def _materials_of(objects):
    mats = set()
    for o in objects:
        if o.type in {"MESH", "CURVE", "FONT"}:
            for s in o.material_slots:
                if s.material:
                    mats.add(s.material)
            if o.data is not None and hasattr(o.data, "materials"):
                for m in o.data.materials:
                    if m:
                        mats.add(m)
    return mats


def _all_objects(col):
    out = list(col.objects)
    for c in col.children:
        out += _all_objects(c)
    return out


# --------------------------------------------------------------------------- #
# the move
# --------------------------------------------------------------------------- #
def poses():
    """(start location, start target, start lens, start shift),
    (end location, end target, end lens, end shift)."""
    m = camera_rig.PRESETS["metro"]
    o = camera_rig.PRESETS["overview"]
    return ((Vector(m["location"]), Vector(m["target"]), m["lens"], m.get("shift_y", 0.0)),
            (Vector(o["location"]), Vector(o["target"]), o["lens"], o.get("shift_y", 0.0)))


def pose_at(u, start, end):
    """Camera pose at eased progress u in [0, 1]."""
    (L0, T0, f0, s0), (L1, T1, f1, s1) = start, end
    D0, D1 = L0 - T0, L1 - T1
    d0, d1 = D0.length, D1.length
    d = d0 * (d1 / d0) ** u                                   # constant ratio per unit u
    k = (d - d1) / (d0 - d1)                                  # 1 -> 0, in step with the distance
    T = T1.lerp(T0, k)
    q0 = Vector((0, 0, 1)).rotation_difference(D0.normalized())
    q1 = Vector((0, 0, 1)).rotation_difference(D1.normalized())
    dirn = (q0.slerp(q1, u) @ Vector((0, 0, 1))).normalized()
    L = T + dirn * d
    lens = f0 + (f1 - f0) * u
    shift = s0 + (s1 - s0) * u
    return L, T, lens, shift


def ground_altitude(L):
    """Height of the camera above the curved ground below it (datum ~ ground)."""
    return L.z - G.drop(L.x, L.y)


def stage(scene, cam, *, metro_cols, site_cols, seconds=12.0, fps=24, hold=1.5,
          fade_hi=9000.0, fade_lo=4500.0):
    """Keyframe the camera and the layer handover. Returns (frames_moving, frame_end)."""
    start, end = poses()
    n_move = max(2, int(round(seconds * fps)))
    n_hold = int(round(hold * fps))
    scene.frame_start = 1
    scene.frame_end = n_move + n_hold
    scene.render.fps = fps

    cam.rotation_mode = "QUATERNION"
    scene[PROP] = 0.0
    mixes = []
    for i in range(n_move):
        f = i + 1
        u = _smoother(i / (n_move - 1))
        L, T, lens, shift = pose_at(u, start, end)
        cam.location = L
        cam.rotation_quaternion = (T - L).to_track_quat("-Z", "Y")
        cam.data.lens = lens
        cam.data.shift_y = shift
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_quaternion", frame=f)
        cam.data.keyframe_insert("lens", frame=f)
        cam.data.keyframe_insert("shift_y", frame=f)
        h = ground_altitude(L)
        mix = _smoothstep(fade_hi, fade_lo, h)
        scene[PROP] = mix
        scene.keyframe_insert(f'["{PROP}"]', frame=f)
        mixes.append(mix)
    # one key per frame already: make them linear so nothing overshoots
    for idb in (cam, cam.data, scene):
        ad = idb.animation_data
        if ad and ad.action:
            for fc in _fcurves(ad.action):
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"

    # hide a layer outright while it is fully faded out
    first_site = next((i + 1 for i, m in enumerate(mixes) if m > 1e-4), n_move)
    last_metro = next((i + 1 for i, m in enumerate(mixes) if m >= 1.0 - 1e-4), n_move)
    for col in site_cols:
        _key_hidden(col, hidden_until=first_site - 1)
    for col in metro_cols:
        _key_hidden(col, hidden_from=last_metro + 1)

    # fade the materials
    site_objs = [o for c in site_cols for o in _all_objects(c)]
    metro_objs = [o for c in metro_cols for o in _all_objects(c)]
    keep = {m for m in _materials_of(metro_objs) if m.name.startswith("metro_ground")}
    for m in _materials_of(site_objs):
        fade_material(m)
    for m in _materials_of(metro_objs) - keep:
        fade_material(m, invert=True)
    # lamps that are real lights (Wilson Hall's floods): energy follows the mix
    for o in site_objs:
        if o.type == "LIGHT":
            e = o.data.energy
            for f in range(1, n_move + 1, 4):
                o.data.energy = e * mixes[f - 1]
                o.data.keyframe_insert("energy", frame=f)
            o.data.energy = e
            o.data.keyframe_insert("energy", frame=n_move)

    h0 = ground_altitude(start[0])
    print(f"[flight] {seconds:.0f}s at {fps} fps = {n_move} moving + {n_hold} held frames; "
          f"altitude {h0 / 1000:.1f} -> {ground_altitude(end[0]) / 1000:.1f} km, "
          f"lens {start[2]:.0f} -> {end[2]:.0f} mm; site layer fades in over frames "
          f"{first_site}-{last_metro} ({fade_hi / 1000:.1f} -> {fade_lo / 1000:.1f} km)")
    return n_move, scene.frame_end


def _fcurves(action):
    from . import common as C
    try:
        return C.action_fcurves(action)
    except Exception:  # noqa: BLE001
        return list(getattr(action, "fcurves", []))


def _key_hidden(col, *, hidden_until=None, hidden_from=None):
    """Keyframe a collection's render visibility (constant interpolation)."""
    if hidden_until is not None and hidden_until >= 1:
        col.hide_render = True
        col.keyframe_insert("hide_render", frame=1)
        col.hide_render = True
        col.keyframe_insert("hide_render", frame=hidden_until)
        col.hide_render = False
        col.keyframe_insert("hide_render", frame=hidden_until + 1)
    if hidden_from is not None:
        col.hide_render = False
        col.keyframe_insert("hide_render", frame=max(1, hidden_from - 1))
        col.hide_render = True
        col.keyframe_insert("hide_render", frame=hidden_from)
