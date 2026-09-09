"""Shared helpers: paths, deterministic RNG, node-graph utilities, PBR
material factory (ambientCG texture sets), and procedural mesh builders.

Everything here is plain `bpy`/`bmesh` and avoids `bpy.ops` so that thousands
of objects can be created quickly in background mode.
"""
from __future__ import annotations

import math
import os
import random

import bmesh
import bpy
from mathutils import Vector

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEX_DIR = os.path.join(ROOT, "assets", "textures")
HDRI_DIR = os.path.join(ROOT, "assets", "hdri")

RNG = random.Random(20260909)  # deterministic scatter / window lights


# --------------------------------------------------------------------------- #
# collections / objects
# --------------------------------------------------------------------------- #
def new_collection(name: str, parent: bpy.types.Collection | None = None) -> bpy.types.Collection:
    col = bpy.data.collections.new(name)
    (parent or bpy.context.scene.collection).children.link(col)
    return col


def link_object(obj: bpy.types.Object, col: bpy.types.Collection | None = None) -> bpy.types.Object:
    (col or bpy.context.scene.collection).objects.link(obj)
    return obj


def mesh_object(name, verts, faces, *, col=None, smooth=False, material=None, location=(0, 0, 0)):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    me.update()
    if smooth:
        me.shade_smooth()
    if material is not None:
        me.materials.append(material)
    obj = bpy.data.objects.new(name, me)
    obj.location = location
    return link_object(obj, col)


def bmesh_object(name, bm, *, col=None, smooth=False, material=None, location=(0, 0, 0), recalc=True):
    if recalc:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.update()
    if smooth:
        me.shade_smooth()
    if material is not None:
        me.materials.append(material)
    obj = bpy.data.objects.new(name, me)
    obj.location = location
    return link_object(obj, col)


def instance(name, mesh, *, col=None, location=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1)):
    """Linked duplicate sharing `mesh` (cheap for thousands of trees/lamps)."""
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    obj.rotation_euler = rotation
    obj.scale = scale
    return link_object(obj, col)


# --------------------------------------------------------------------------- #
# node helpers
# --------------------------------------------------------------------------- #
def sock_in(node, identifier):
    """Input socket by *identifier* (names are duplicated on Mix/Math nodes)."""
    for s in node.inputs:
        if s.identifier == identifier:
            return s
    return node.inputs[identifier]


def sock_out(node, identifier):
    for s in node.outputs:
        if s.identifier == identifier:
            return s
    return node.outputs[identifier]


def node(tree, type_, **props):
    n = tree.nodes.new(type_)
    for k, v in props.items():
        setattr(n, k, v)
    return n


def link(tree, a, b):
    """Link two sockets (or node.outputs[x] -> node.inputs[y])."""
    tree.links.new(a, b)


def nmath(tree, op, a=None, b=None, *, value_a=None, value_b=None, clamp=False):
    n = tree.nodes.new("ShaderNodeMath")
    n.operation = op
    n.use_clamp = clamp
    if a is not None:
        tree.links.new(a, n.inputs[0])
    elif value_a is not None:
        n.inputs[0].default_value = value_a
    if b is not None:
        tree.links.new(b, n.inputs[1])
    elif value_b is not None:
        n.inputs[1].default_value = value_b
    return n


def mix_color(tree, fac, a, b, blend="MIX"):
    """Mix node (RGBA). `fac`, `a`, `b` may be sockets or constant values."""
    n = tree.nodes.new("ShaderNodeMix")
    n.data_type = "RGBA"
    n.blend_type = blend
    for target, val in ((sock_in(n, "Factor_Float"), fac), (sock_in(n, "A_Color"), a), (sock_in(n, "B_Color"), b)):
        if isinstance(val, bpy.types.NodeSocket):
            tree.links.new(val, target)
        elif isinstance(val, (int, float)):
            target.default_value = val
        else:
            target.default_value = tuple(val) if len(val) == 4 else (*val, 1.0)
    return n, sock_out(n, "Result_Color")


def mix_float(tree, fac, a, b):
    n = tree.nodes.new("ShaderNodeMix")
    n.data_type = "FLOAT"
    for target, val in ((sock_in(n, "Factor_Float"), fac), (sock_in(n, "A_Float"), a), (sock_in(n, "B_Float"), b)):
        if isinstance(val, bpy.types.NodeSocket):
            tree.links.new(val, target)
        else:
            target.default_value = val
    return n, sock_out(n, "Result_Float")


def blackbody(tree, kelvin):
    n = tree.nodes.new("ShaderNodeBlackbody")
    if isinstance(kelvin, bpy.types.NodeSocket):
        tree.links.new(kelvin, n.inputs["Temperature"])
    else:
        n.inputs["Temperature"].default_value = kelvin
    return n.outputs["Color"]


def kelvin_rgb(kelvin: float) -> tuple[float, float, float]:
    """Approximate blackbody colour (for materials; lights use the native control)."""
    t = kelvin / 100.0
    r = 255.0 if t <= 66 else 329.698727446 * ((t - 60) ** -0.1332047592)
    g = (99.4708025861 * math.log(t) - 161.1195681661) if t <= 66 else 288.1221695283 * ((t - 60) ** -0.0755148492)
    if t >= 66:
        b = 255.0
    elif t <= 19:
        b = 0.0
    else:
        b = 138.5177312231 * math.log(t - 10) - 305.0447927307
    c = lambda x: max(0.0, min(255.0, x)) / 255.0  # noqa: E731
    return c(r), c(g), c(b)


# --------------------------------------------------------------------------- #
# materials
# --------------------------------------------------------------------------- #
_IMG_CACHE: dict[tuple[str, bool], bpy.types.Image] = {}


def load_image(path: str, non_color: bool = False) -> bpy.types.Image:
    key = (path, non_color)
    if key not in _IMG_CACHE:
        img = bpy.data.images.load(path, check_existing=True)
        if non_color:
            img.colorspace_settings.name = "Non-Color"
        _IMG_CACHE[key] = img
    return _IMG_CACHE[key]


def tex_set(name: str) -> dict[str, str]:
    """Map ambientCG map names -> file paths for a downloaded texture set."""
    d = os.path.join(TEX_DIR, name)
    maps: dict[str, str] = {}
    if not os.path.isdir(d):
        return maps
    for f in os.listdir(d):
        for key in ("Color", "Roughness", "NormalGL", "Displacement", "AmbientOcclusion"):
            if f.endswith(f"_{key}.jpg"):
                maps[key] = os.path.join(d, f)
    return maps


def new_material(name: str) -> tuple[bpy.types.Material, bpy.types.NodeTree, bpy.types.Node, bpy.types.Node]:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    return mat, nt, nt.nodes["Principled BSDF"], nt.nodes["Material Output"]


def flat_material(name, color, *, roughness=0.6, metallic=0.0, emission=None, emission_strength=0.0):
    mat, nt, bsdf, _ = new_material(name)
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if emission is not None:
        bsdf.inputs["Emission Color"].default_value = (*emission, 1.0)
        bsdf.inputs["Emission Strength"].default_value = emission_strength
    return mat


def emissive_material(name, color, strength, *, camera_strength=None):
    """Pure emitter. `camera_strength` lets the visible core differ from the
    light it casts (Light Path > Is Camera Ray), which keeps glowing tubes
    readable without blowing out the fog around them."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.remove(nt.nodes["Principled BSDF"])
    out = nt.nodes["Material Output"]
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*color, 1.0)
    if camera_strength is None:
        em.inputs["Strength"].default_value = strength
    else:
        lp = nt.nodes.new("ShaderNodeLightPath")
        _, s = mix_float(nt, lp.outputs["Is Camera Ray"], strength, camera_strength)
        nt.links.new(s, em.inputs["Strength"])
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat


def pbr_material(
    name: str,
    texset: str,
    *,
    scale: float = 4.0,
    tint=(1.0, 1.0, 1.0),
    roughness_mult: float = 1.0,
    normal_strength: float = 1.0,
    projection: str = "BOX",
    coords: str = "Object",
    fallback_color=(0.3, 0.3, 0.3),
):
    """Principled material driven by an ambientCG set (Color/Roughness/NormalGL).

    Uses object-space box projection so any procedurally built mesh is textured
    without UVs. `scale` is the physical tile size in metres.
    """
    mat, nt, bsdf, _ = new_material(name)
    maps = tex_set(texset)
    if "Color" not in maps:  # assets not fetched: degrade gracefully
        bsdf.inputs["Base Color"].default_value = (*[c * t for c, t in zip(fallback_color, tint)], 1.0)
        bsdf.inputs["Roughness"].default_value = 0.8 * roughness_mult
        return mat

    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1.0 / scale, 1.0 / scale, 1.0 / scale)
    nt.links.new(tc.outputs[coords], mp.inputs["Vector"])

    def img(path, non_color):
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = load_image(path, non_color)
        n.projection = projection
        if projection == "BOX":
            n.projection_blend = 0.3
        n.extension = "REPEAT"
        nt.links.new(mp.outputs["Vector"], n.inputs["Vector"])
        return n

    col = img(maps["Color"], False)
    _, tinted = mix_color(nt, 1.0, col.outputs["Color"], (*tint, 1.0), "MULTIPLY")
    nt.links.new(tinted, bsdf.inputs["Base Color"])

    if "Roughness" in maps:
        r = img(maps["Roughness"], True)
        m = nmath(nt, "MULTIPLY", r.outputs["Color"], value_b=roughness_mult, clamp=True)
        nt.links.new(m.outputs[0], bsdf.inputs["Roughness"])
    else:
        bsdf.inputs["Roughness"].default_value = 0.8 * roughness_mult

    if "NormalGL" in maps and normal_strength > 0:
        nrm = img(maps["NormalGL"], True)
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.inputs["Strength"].default_value = normal_strength
        nt.links.new(nrm.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def water_material(name="water", *, tint=(0.010, 0.020, 0.028), ripple_scale=3.0, ripple_strength=0.06):
    """Dark still water: Fresnel reflections from IOR 1.33, gentle procedural ripples."""
    mat, nt, bsdf, _ = new_material(name)
    bsdf.inputs["Base Color"].default_value = (*tint, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.035
    bsdf.inputs["IOR"].default_value = 1.333
    bsdf.inputs["Specular IOR Level"].default_value = 0.5
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1.0 / ripple_scale, 1.0 / (ripple_scale * 2.2), 1.0)  # wind-stretched
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 1.0
    noise.inputs["Detail"].default_value = 4.0
    noise.inputs["Roughness"].default_value = 0.55
    nt.links.new(mp.outputs["Vector"], noise.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = ripple_strength
    bump.inputs["Distance"].default_value = 0.2
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


# --------------------------------------------------------------------------- #
# procedural meshes
# --------------------------------------------------------------------------- #
def ring_mesh(name, center, radius, profile, *, segments=360, ry=None, col=None, material=None, smooth=True, a0=0.0, a1=None):
    """Sweep a 2D profile [(radial offset, z), ...] around a circle/ellipse.

    `radius` is the x semi-axis; `ry` the y semi-axis (defaults to circle).
    A partial sweep is produced when `a1` is given (radians).
    """
    ry = radius if ry is None else ry
    closed = a1 is None
    a1 = 2 * math.pi if a1 is None else a1
    bm = bmesh.new()
    rows = []
    n = segments if closed else segments + 1
    for i in range(n):
        a = a0 + (a1 - a0) * i / segments
        ca, sa = math.cos(a), math.sin(a)
        row = []
        for dr, dz in profile:
            x = center[0] + (radius + dr) * ca
            y = center[1] + (ry + dr) * sa
            row.append(bm.verts.new((x, y, dz)))
        rows.append(row)
    for i in range(segments):
        r0, r1 = rows[i], rows[(i + 1) % n]
        for j in range(len(profile) - 1):
            bm.faces.new((r0[j], r1[j], r1[j + 1], r0[j + 1]))
    return bmesh_object(name, bm, col=col, smooth=smooth, material=material)


def tube_mesh(name, points, radius, *, sides=10, col=None, material=None, closed=False):
    """Circular tube along a 3D polyline (used for the collider beamline)."""
    bm = bmesh.new()
    pts = [Vector(p) for p in points]
    n = len(pts)
    rings = []
    for i, p in enumerate(pts):
        if closed:
            t = (pts[(i + 1) % n] - pts[i - 1]).normalized()
        else:
            t = (pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]).normalized()
        up = Vector((0, 0, 1)) if abs(t.z) < 0.9 else Vector((1, 0, 0))
        u = t.cross(up).normalized()
        v = u.cross(t).normalized()
        rings.append([bm.verts.new(p + radius * (math.cos(2 * math.pi * k / sides) * u + math.sin(2 * math.pi * k / sides) * v)) for k in range(sides)])
    m = n if closed else n - 1
    for i in range(m):
        r0, r1 = rings[i], rings[(i + 1) % n]
        for k in range(sides):
            bm.faces.new((r0[k], r1[k], r1[(k + 1) % sides], r0[(k + 1) % sides]))
    return bmesh_object(name, bm, col=col, smooth=True, material=material)


def ribbon_mesh(name, points, width, z, *, col=None, material=None, closed=False):
    """Flat strip of constant width along a 2D polyline (roads, paths)."""
    bm = bmesh.new()
    pts = [Vector((p[0], p[1], 0.0)) for p in points]
    n = len(pts)
    left, right = [], []
    for i, p in enumerate(pts):
        if closed:
            t = (pts[(i + 1) % n] - pts[i - 1]).normalized()
        else:
            t = (pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]).normalized()
        nrm = Vector((-t.y, t.x, 0.0)) * (width / 2)
        left.append(bm.verts.new((p.x + nrm.x, p.y + nrm.y, z)))
        right.append(bm.verts.new((p.x - nrm.x, p.y - nrm.y, z)))
    m = n if closed else n - 1
    for i in range(m):
        j = (i + 1) % n
        bm.faces.new((left[i], right[i], right[j], left[j]))
    return bmesh_object(name, bm, col=col, material=material)


def circle_points(center, r, n=256, ry=None, a0=0.0, a1=2 * math.pi, z=None):
    ry = r if ry is None else ry
    pts = []
    for i in range(n):
        a = a0 + (a1 - a0) * i / n
        p = (center[0] + r * math.cos(a), center[1] + ry * math.sin(a))
        pts.append(p if z is None else (*p, z))
    return pts


def box_bmesh(sx, sy, sz, *, origin_bottom=True):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(sx, sy, sz), verts=bm.verts)
    if origin_bottom:
        bmesh.ops.translate(bm, vec=(0, 0, sz / 2), verts=bm.verts)
    return bm


def box_object(name, size, location, *, rot_z=0.0, col=None, material=None):
    bm = box_bmesh(*size)
    obj = bmesh_object(name, bm, col=col, material=material, location=location)
    obj.rotation_euler = (0, 0, rot_z)
    return obj


def box_mesh_data(name="unit_box"):
    """Shared unit cube (bottom at z=0) for instanced buildings."""
    bm = box_bmesh(1, 1, 1)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    return me


def disk_mesh(name, center, r, *, z=0.0, n=96, ry=None, col=None, material=None, rot=0.0):
    ry = r if ry is None else ry
    bm = bmesh.new()
    vs = []
    for i in range(n):
        a = 2 * math.pi * i / n
        x, y = r * math.cos(a), ry * math.sin(a)
        xr = x * math.cos(rot) - y * math.sin(rot)
        yr = x * math.sin(rot) + y * math.cos(rot)
        vs.append(bm.verts.new((center[0] + xr, center[1] + yr, z)))
    bm.faces.new(vs)
    return bmesh_object(name, bm, col=col, material=material)


def blob_mesh(name, center, r, *, z=0.0, n=96, ry=None, wobble=0.18, seed=0, col=None, material=None, rot=0.0):
    """Irregular pond/lake outline: ellipse with low-frequency radial noise."""
    rng = random.Random(seed)
    ry = r if ry is None else ry
    k = [rng.uniform(-1, 1) for _ in range(6)]
    bm = bmesh.new()
    vs = []
    for i in range(n):
        a = 2 * math.pi * i / n
        w = 1.0 + wobble * (0.6 * k[0] * math.sin(2 * a + k[1] * 3) + 0.3 * k[2] * math.sin(3 * a + k[3] * 3) + 0.25 * k[4] * math.sin(5 * a + k[5] * 3))
        x, y = r * w * math.cos(a), ry * w * math.sin(a)
        xr = x * math.cos(rot) - y * math.sin(rot)
        yr = x * math.sin(rot) + y * math.cos(rot)
        vs.append(bm.verts.new((center[0] + xr, center[1] + yr, z)))
    bm.faces.new(vs)
    return bmesh_object(name, bm, col=col, material=material)


def cylinder_bmesh(r, h, *, n=48, cap=True):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=cap, segments=n, radius1=r, radius2=r, depth=h)
    bmesh.ops.translate(bm, vec=(0, 0, h / 2), verts=bm.verts)
    return bm


def cone_bmesh(r0, r1, h, *, n=3, cap=True):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=cap, segments=n, radius1=r0, radius2=r1, depth=h)
    bmesh.ops.translate(bm, vec=(0, 0, h / 2), verts=bm.verts)
    return bm


def canopy_mesh_data(name, seed, *, subdiv=2, lumpiness=0.28):
    """Lumpy icosphere used as a tree crown (instanced, so a few variants suffice)."""
    rng = random.Random(seed)
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=1.0)
    for v in bm.verts:
        v.co *= 1.0 + lumpiness * (rng.random() - 0.5)
        v.co.z *= 1.15
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.shade_smooth()
    return me


def sphere_mesh_data(name, r, *, subdiv=1):
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=r)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.shade_smooth()
    return me


# --------------------------------------------------------------------------- #
# lights
# --------------------------------------------------------------------------- #
def aim(obj: bpy.types.Object, target) -> None:
    d = Vector(target) - Vector(obj.location)
    obj.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def spot_light(name, location, target, *, power, kelvin=3000, size_deg=50, blend=0.5, radius=0.5, col=None):
    data = bpy.data.lights.new(name, "SPOT")
    data.energy = power
    data.use_temperature = True
    data.temperature = kelvin
    data.spot_size = math.radians(size_deg)
    data.spot_blend = blend
    data.shadow_soft_size = radius
    obj = bpy.data.objects.new(name, data)
    obj.location = location
    link_object(obj, col)
    aim(obj, target)
    return obj


def point_light(name, location, *, power, kelvin=3000, radius=1.0, col=None):
    data = bpy.data.lights.new(name, "POINT")
    data.energy = power
    data.use_temperature = True
    data.temperature = kelvin
    data.shadow_soft_size = radius
    obj = bpy.data.objects.new(name, data)
    obj.location = location
    return link_object(obj, col)


def area_light(name, location, target, *, power, size, kelvin=3200, col=None, shape="DISK"):
    data = bpy.data.lights.new(name, "AREA")
    data.shape = shape
    data.size = size
    data.energy = power
    data.use_temperature = True
    data.temperature = kelvin
    obj = bpy.data.objects.new(name, data)
    obj.location = location
    link_object(obj, col)
    aim(obj, target)
    return obj


def sun_light(name, azimuth_deg, elevation_deg, *, energy, kelvin=4300, angle_deg=0.53, col=None):
    """Directional light coming *from* the given azimuth (deg from +Y/north,
    clockwise) and elevation."""
    data = bpy.data.lights.new(name, "SUN")
    data.energy = energy
    data.angle = math.radians(angle_deg)
    data.use_temperature = True
    data.temperature = kelvin
    obj = bpy.data.objects.new(name, data)
    link_object(obj, col)
    d = direction_from(azimuth_deg, elevation_deg)
    obj.location = d * 1000.0
    aim(obj, (0, 0, 0))
    return obj


def direction_from(azimuth_deg: float, elevation_deg: float) -> Vector:
    """Unit vector pointing *toward* a sky position (azimuth from north, clockwise)."""
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    return Vector((math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)))
