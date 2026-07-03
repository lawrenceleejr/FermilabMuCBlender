"""Build the Fermilab muon collider scene inside Blender.

Run headlessly:

    blender --background --factory-startup --python scripts/build_scene.py -- \
        --style schematic --out out/fermilab_muc_schematic.blend [--fast]

Reads only the committed snapshot under data/ (no network).
"""

import argparse
import json
import math
import os
import sys

import bpy
import bmesh
import numpy as np
from mathutils import Vector
from mathutils.geometry import tessellate_polygon

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from config import facility  # noqa: E402

DATA = os.path.join(REPO, "data")

# Cutaway: remove this quadrant of terrain (in ring-center frame, out to
# CUT_RADIUS) so the underground complex is visible. +x east, -y south =>
# southeast quarter-disk pit.
CUT_QUADRANT = (1, -1)
CUT_RADIUS = 3300.0


def in_cut(x, y):
    dx = x - facility.CAMPUS_CENTER[0]
    dy = y - facility.CAMPUS_CENTER[1]
    return (math.copysign(1, dx) == CUT_QUADRANT[0]
            and math.copysign(1, dy) == CUT_QUADRANT[1]
            and dx * dx + dy * dy < CUT_RADIUS * CUT_RADIUS)


# --- generic helpers ---------------------------------------------------------

def collection(name):
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    c = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(c)
    return c


def link_to(obj, coll_name):
    for c in obj.users_collection:
        c.objects.unlink(obj)
    collection(coll_name).objects.link(obj)


def emission_material(name, color, strength=3.0, mix_principled=0.0):
    """Emissive color-coded material; optionally mixed with a Principled BSDF."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*color, 1.0)
    em.inputs["Strength"].default_value = strength
    if mix_principled > 0.0:
        pr = nt.nodes.new("ShaderNodeBsdfPrincipled")
        pr.inputs["Base Color"].default_value = (*color, 1.0)
        pr.inputs["Roughness"].default_value = 0.4
        mix = nt.nodes.new("ShaderNodeMixShader")
        mix.inputs["Fac"].default_value = mix_principled
        nt.links.new(em.outputs["Emission"], mix.inputs[1])
        nt.links.new(pr.outputs["BSDF"], mix.inputs[2])
        nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    else:
        nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    mat.diffuse_color = (*color, 1.0)
    return mat


def principled_material(name, color, roughness=0.9, alpha=1.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    if alpha < 1.0:
        bsdf.inputs["Alpha"].default_value = alpha
    mat.diffuse_color = (*color, alpha)
    return mat


# --- heightmap ----------------------------------------------------------------

class Heightmap:
    def __init__(self, path):
        d = np.load(path)
        self.z = d["z"].astype(np.float64) - float(d["z_datum"])
        self.x0 = float(d["x0"])
        self.y0 = float(d["y0"])
        self.dx = float(d["dx"])
        self.dy = float(d["dy"])
        self.n = self.z.shape[0]
        self.xext = (self.n - 1) * self.dx
        self.yext = (self.n - 1) * self.dy

    def sample(self, x, y):
        """Bilinear ground height at ENU (x, y); scene z (datum-relative)."""
        fx = (x - self.x0) / self.dx
        fy = (y - self.y0) / self.dy
        fx = min(max(fx, 0.0), self.n - 1.001)
        fy = min(max(fy, 0.0), self.n - 1.001)
        i, j = int(fy), int(fx)
        ax, ay = fx - j, fy - i
        z = self.z
        return float(z[i, j] * (1 - ax) * (1 - ay) + z[i, j + 1] * ax * (1 - ay)
                     + z[i + 1, j] * (1 - ax) * ay + z[i + 1, j + 1] * ax * ay)


# --- terrain -------------------------------------------------------------------

def build_terrain(hm, style, cutaway):
    """Grid mesh from the heightmap. Realistic style: solid block with an
    optional quadrant removed (numpy mask — no booleans). Schematic style:
    flat translucent plane."""
    if style == "schematic":
        size = 22000.0
        cx, cy = facility.CAMPUS_CENTER
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(cx, cy, 0.0))
        obj = bpy.context.object
        obj.scale = (size, size, 1.0)
        obj.name = "Terrain_Schematic"
        mat = principled_material("GroundSchematic", (0.045, 0.055, 0.07),
                                  alpha=0.72)
        mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 1.0
        obj.data.materials.append(mat)
        link_to(obj, "Terrain")
        return obj

    stride = 1  # full 512 grid; the cut rim needs the resolution
    z = hm.z[::stride, ::stride]
    n = z.shape[0]
    xs = hm.x0 + np.arange(0, hm.n, stride) * hm.dx
    ys = hm.y0 + np.arange(0, hm.n, stride) * hm.dy
    X, Y = np.meshgrid(xs, ys)

    keep = np.ones((n - 1, n - 1), dtype=bool)  # per-quad mask
    if cutaway:
        cx, cy = facility.CAMPUS_CENTER
        qx = (X[:-1, :-1] + X[1:, 1:]) / 2 - cx
        qy = (Y[:-1, :-1] + Y[1:, 1:]) / 2 - cy
        keep &= ~((np.sign(qx) == CUT_QUADRANT[0])
                  & (np.sign(qy) == CUT_QUADRANT[1])
                  & (qx * qx + qy * qy < CUT_RADIUS * CUT_RADIUS))

    verts_top = np.stack([X.ravel(), Y.ravel(), z.ravel()], axis=1)
    zb = facility.TERRAIN_SOLID_BOTTOM
    verts_bot = verts_top.copy()
    verts_bot[:, 2] = zb
    verts = np.concatenate([verts_top, verts_bot])
    nv = n * n

    def vid(i, j, bottom=False):
        return i * n + j + (nv if bottom else 0)

    faces = []
    ii, jj = np.nonzero(keep)
    for i, j in zip(ii.tolist(), jj.tolist()):
        faces.append((vid(i, j), vid(i, j + 1), vid(i + 1, j + 1), vid(i + 1, j)))
    # side walls wherever a kept quad borders a removed/outside quad
    padded = np.zeros((n + 1, n + 1), dtype=bool)
    padded[1:n, 1:n] = keep
    for i, j in zip(ii.tolist(), jj.tolist()):
        pi, pj = i + 1, j + 1
        if not padded[pi - 1, pj]:  # south edge exposed
            faces.append((vid(i, j), vid(i, j + 1),
                          vid(i, j + 1, True), vid(i, j, True)))
        if not padded[pi + 1, pj]:  # north edge
            faces.append((vid(i + 1, j + 1), vid(i + 1, j),
                          vid(i + 1, j, True), vid(i + 1, j + 1, True)))
        if not padded[pi, pj - 1]:  # west edge
            faces.append((vid(i + 1, j), vid(i, j),
                          vid(i, j, True), vid(i + 1, j, True)))
        if not padded[pi, pj + 1]:  # east edge
            faces.append((vid(i, j + 1), vid(i + 1, j + 1),
                          vid(i + 1, j + 1, True), vid(i, j + 1, True)))
    # bottom faces for every quad: block underside + pit floor in the cut
    for i in range(n - 1):
        for j in range(n - 1):
            faces.append((vid(i + 1, j, True), vid(i + 1, j + 1, True),
                          vid(i, j + 1, True), vid(i, j, True)))

    mesh = bpy.data.meshes.new("TerrainMesh")
    mesh.from_pydata(verts.tolist(), [], faces)
    mesh.validate()
    obj = bpy.data.objects.new("Terrain", mesh)
    bpy.context.scene.collection.objects.link(obj)
    link_to(obj, "Terrain")

    # planar-XY UV layer for the imagery drape (vectorized; ~2M loops)
    uvl = mesh.uv_layers.new(name="UVMap")
    nloops = len(mesh.loops)
    vidx = np.empty(nloops, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", vidx)
    co = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    uv = np.empty((nloops, 2))
    uv[:, 0] = (co[vidx, 0] - hm.x0) / hm.xext
    uv[:, 1] = (co[vidx, 1] - hm.y0) / hm.yext
    uvl.data.foreach_set("uv", uv.ravel())

    # Material: green-brown by height above datum on top; stratigraphy
    # stripes on the cut faces (driven by world z below ~-2 m).
    mat = bpy.data.materials.new("TerrainRealistic")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.95
    geom = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geom.outputs["Position"], sep.inputs["Vector"])
    # surface branch: draped aerial imagery if available, else elevation tint
    img_path = os.path.join(DATA, "imagery.jpg")
    if os.path.exists(img_path):
        img = bpy.data.images.load(img_path, check_existing=True)
        img.pack()  # self-contained .blend
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = img
        tex.interpolation = "Linear"
        tex.extension = "EXTEND"
        uvn = nt.nodes.new("ShaderNodeUVMap")
        uvn.uv_map = "UVMap"
        nt.links.new(uvn.outputs["UV"], tex.inputs["Vector"])
        hsv = nt.nodes.new("ShaderNodeHueSaturation")
        hsv.inputs["Saturation"].default_value = 1.15  # counter Filmic wash-out
        nt.links.new(tex.outputs["Color"], hsv.inputs["Color"])
        surf_color_out = hsv.outputs["Color"]
    else:
        print("terrain: data/imagery.jpg missing -> elevation tint fallback")
        ramp_surf = nt.nodes.new("ShaderNodeValToRGB")
        ramp_surf.color_ramp.elements[0].position = 0.0
        ramp_surf.color_ramp.elements[0].color = (0.06, 0.18, 0.04, 1.0)
        ramp_surf.color_ramp.elements[1].position = 1.0
        ramp_surf.color_ramp.elements[1].color = (0.28, 0.24, 0.11, 1.0)
        map_surf = nt.nodes.new("ShaderNodeMapRange")
        map_surf.inputs["From Min"].default_value = -20.0
        map_surf.inputs["From Max"].default_value = 25.0
        nt.links.new(sep.outputs["Z"], map_surf.inputs["Value"])
        nt.links.new(map_surf.outputs["Result"], ramp_surf.inputs["Fac"])
        surf_color_out = ramp_surf.outputs["Color"]
    # underground stratigraphy: till above GLACIAL_TILL_BOTTOM, dolomite below
    ramp_geo = nt.nodes.new("ShaderNodeValToRGB")
    ramp_geo.color_ramp.interpolation = "CONSTANT"
    e0 = ramp_geo.color_ramp.elements[0]
    e0.position = 0.0
    e0.color = (0.16, 0.19, 0.24, 1.0)  # dolomite gray-blue (deep)
    e1 = ramp_geo.color_ramp.elements[1]
    e1.position = (facility.GLACIAL_TILL_BOTTOM - facility.TERRAIN_SOLID_BOTTOM) \
        / (0.0 - facility.TERRAIN_SOLID_BOTTOM)
    e1.color = (0.30, 0.22, 0.12, 1.0)  # glacial till tan (shallow)
    map_geo = nt.nodes.new("ShaderNodeMapRange")
    map_geo.inputs["From Min"].default_value = facility.TERRAIN_SOLID_BOTTOM
    map_geo.inputs["From Max"].default_value = 0.0
    nt.links.new(sep.outputs["Z"], map_geo.inputs["Value"])
    nt.links.new(map_geo.outputs["Result"], ramp_geo.inputs["Fac"])
    # surface = upward-facing AND not deep (near-vertical pit walls and the
    # pit floor / block underside stay on the stratigraphy branch)
    sep_n = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geom.outputs["Normal"], sep_n.inputs["Vector"])
    up_test = nt.nodes.new("ShaderNodeMath")
    up_test.operation = "GREATER_THAN"
    up_test.inputs[1].default_value = 0.7
    nt.links.new(sep_n.outputs["Z"], up_test.inputs[0])
    z_test = nt.nodes.new("ShaderNodeMath")
    z_test.operation = "GREATER_THAN"
    z_test.inputs[1].default_value = -40.0
    nt.links.new(sep.outputs["Z"], z_test.inputs[0])
    both = nt.nodes.new("ShaderNodeMath")
    both.operation = "MULTIPLY"
    nt.links.new(up_test.outputs["Value"], both.inputs[0])
    nt.links.new(z_test.outputs["Value"], both.inputs[1])
    mixc = nt.nodes.new("ShaderNodeMix")
    mixc.data_type = "RGBA"
    nt.links.new(both.outputs["Value"], mixc.inputs["Factor"])
    nt.links.new(ramp_geo.outputs["Color"], mixc.inputs[6])  # A: underground
    nt.links.new(surf_color_out, mixc.inputs[7])             # B: surface
    # faint emission so the pit interior isn't pitch black in shadow
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = 0.05
    nt.links.new(mixc.outputs[2], em.inputs["Color"])
    add = nt.nodes.new("ShaderNodeAddShader")
    out_node = nt.nodes["Material Output"]
    nt.links.new(bsdf.outputs["BSDF"], add.inputs[0])
    nt.links.new(em.outputs["Emission"], add.inputs[1])
    nt.links.new(add.outputs["Shader"], out_node.inputs["Surface"])
    nt.links.new(mixc.outputs[2], bsdf.inputs["Base Color"])
    obj.data.materials.append(mat)

    # apron planes just below grade around the terrain block, so its edges
    # don't read as a 260 m cliff from low cameras (they can't cross the pit,
    # which sits well inside the grid)
    apron_mat = principled_material("ApronMat", (0.13, 0.17, 0.09),
                                    roughness=1.0)
    x0, y0 = hm.x0, hm.y0
    x1, y1 = hm.x0 + hm.xext, hm.y0 + hm.yext
    reach = 30000.0
    az = -25.0
    aprons = [
        ((x0 + x1) / 2, y1 + reach / 2, hm.xext + 2 * reach, reach),  # north
        ((x0 + x1) / 2, y0 - reach / 2, hm.xext + 2 * reach, reach),  # south
        (x1 + reach / 2, (y0 + y1) / 2, reach, hm.yext),              # east
        (x0 - reach / 2, (y0 + y1) / 2, reach, hm.yext),              # west
    ]
    for i, (cx_, cy_, sx, sy) in enumerate(aprons):
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(cx_, cy_, az))
        ap = bpy.context.object
        ap.scale = (sx, sy, 1.0)
        ap.name = f"Apron_{i}"
        ap.data.materials.append(apron_mat)
        link_to(ap, "Terrain")
    return obj


def build_boundary(hm, style):
    with open(os.path.join(DATA, "site_boundary.geojson")) as f:
        ring = json.load(f)["features"][0]["geometry"]["coordinates"][0]
    curve = bpy.data.curves.new("SiteBoundary", type="CURVE")
    curve.dimensions = "3D"
    sp = curve.splines.new("POLY")
    sp.points.add(len(ring) - 1)
    for p, (x, y) in zip(sp.points, ring):
        z = 2.0 if style == "schematic" else hm.sample(x, y) + 2.0
        p.co = (x, y, z, 1.0)
    sp.use_cyclic_u = True
    curve.bevel_depth = 8.0
    curve.bevel_resolution = 2
    obj = bpy.data.objects.new("SiteBoundary", curve)
    bpy.context.scene.collection.objects.link(obj)
    bcol = (0.75, 0.78, 0.85) if style == "schematic" else (0.12, 0.12, 0.15)
    obj.data.materials.append(
        emission_material("BoundaryMat", bcol, strength=1.5))
    link_to(obj, "Terrain")


# --- water ---------------------------------------------------------------------

def build_water(hm, cutaway):
    """Flat specular caps for OSM water polygons (realistic style only)."""
    path = os.path.join(DATA, "water.geojson")
    if not os.path.exists(path):
        print("water: data/water.geojson missing, skipping")
        return
    with open(path) as f:
        feats = json.load(f).get("features", [])
    if not feats:
        print("water: empty collection, skipping")
        return

    bm = bmesh.new()
    placed = 0
    for feat in feats:
        ring = feat["geometry"]["coordinates"][0]
        ring = [p for i, p in enumerate(ring)
                if i == 0 or (abs(p[0] - ring[i - 1][0]) > 1e-6
                              or abs(p[1] - ring[i - 1][1]) > 1e-6)]
        if len(ring) < 3:
            continue
        if cutaway and any(in_cut(x, y) for x, y in ring):
            continue
        # drop polygons leaking past the terrain grid (e.g. Fox River)
        if any(not (hm.x0 - 100 < x < hm.x0 + hm.xext + 100
                    and hm.y0 - 100 < y < hm.y0 + hm.yext + 100)
               for x, y in ring):
            continue
        coords = [Vector((x, y, 0.0)) for x, y in ring]
        tris = tessellate_polygon([coords])
        if not tris:
            continue
        z = (sum(hm.sample(x, y) for x, y in ring) / len(ring)
             + facility.WATER_Z_OFFSET)
        vs = [bm.verts.new((x, y, z)) for x, y in ring]
        for a, b, c in tris:
            try:
                bm.faces.new((vs[a], vs[b], vs[c]))
            except ValueError:
                pass
        placed += 1

    mesh = bpy.data.meshes.new("WaterMesh")
    bm.to_mesh(mesh)
    bm.free()
    mesh.validate()
    obj = bpy.data.objects.new("Water", mesh)
    bpy.context.scene.collection.objects.link(obj)
    mat = bpy.data.materials.new("WaterMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.02, 0.05, 0.09, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.08
    obj.data.materials.append(mat)
    link_to(obj, "Terrain")
    print(f"water: {placed} polygons placed")


# --- Wilson Hall sculpted model --------------------------------------------------

def _add_box(bm, center, size, slot_map, slot):
    cx, cy, cz = center
    sx, sy, sz = size[0] / 2, size[1] / 2, size[2] / 2
    v = [bm.verts.new((cx + dx * sx, cy + dy * sy, cz + dz * sz))
         for dx, dy, dz in ((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                            (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))]
    for idx in ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
                (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3)):
        slot_map[bm.faces.new([v[i] for i in idx])] = slot


def build_wilson_hall(hm, style):
    """Sculpted twin-tower model: per-floor lofted cross-sections with a
    quadratic inward sweep, atrium gap, crossover bridges, crown slab.
    Material slots: 0 glass curtain walls, 1 concrete, 2 roof."""
    P = facility.WILSON_HALL_MODEL
    H = P["floors"] * P["floor_h"]
    L = P["length"]

    def profiles(z):
        t = z / H
        outer = (P["half_width_top"]
                 + (P["half_width_base"] - P["half_width_top"]) * (1 - t) ** 2)
        gap = max(P["gap_half_min"], P["gap_half_base"] * (1 - t) ** 1.2)
        return outer, gap

    bm = bmesh.new()
    slot_map = {}
    for sign in (1.0, -1.0):
        rings = []
        for k in range(P["floors"] + 1):
            z = k * P["floor_h"]
            outer, gap = profiles(z)
            y0, y1 = sign * gap, sign * outer
            rings.append([bm.verts.new(v) for v in
                          ((-L / 2, y0, z), (L / 2, y0, z),
                           (L / 2, y1, z), (-L / 2, y1, z))])
        for k in range(P["floors"]):
            a, b = rings[k], rings[k + 1]
            for e in range(4):
                f = bm.faces.new((a[e], a[(e + 1) % 4],
                                  b[(e + 1) % 4], b[e]))
                # edges 0 (inner y0-y0) and 2 (outer y1-y1) are the big
                # curtain-wall faces; 1 and 3 are the +-x concrete ends
                slot_map[f] = 0 if e in (0, 2) else 1
        slot_map[bm.faces.new(rings[0])] = 1
        slot_map[bm.faces.new(list(reversed(rings[-1])))] = 2

    for k in P["bridge_floors"]:
        z = k * P["floor_h"]
        _, gap = profiles(z)
        _add_box(bm, (0.0, 0.0, z + 1.6),
                 (P["bridge_width"], 2 * gap + 1.2, 3.2), slot_map, 1)
    outer_top, _ = profiles(H)
    _add_box(bm, (0.0, 0.0, H + 0.9), (L, 2 * outer_top, 1.8), slot_map, 2)

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mat_idx = [slot_map.get(f, 1) for f in bm.faces]
    mesh = bpy.data.meshes.new("WilsonHallMesh")
    bm.to_mesh(mesh)
    bm.free()
    mesh.validate()
    mesh.polygons.foreach_set("material_index", mat_idx)

    obj = bpy.data.objects.new("WilsonHall", mesh)
    obj.rotation_euler = (0.0, 0.0, math.radians(90.0 - P["rotation_deg"]))
    base = 0.0 if style == "schematic" else hm.sample(0.0, 0.0) - 1.0
    obj.location = (0.0, 0.0, base)
    bpy.context.scene.collection.objects.link(obj)
    link_to(obj, "Buildings")

    if style == "schematic":
        m = emission_material("WilsonHallMat", (0.00, 0.45, 0.70),
                              strength=1.0, mix_principled=0.75)
        for _ in range(3):
            obj.data.materials.append(m)
        return

    # slot 0: glass curtain wall with procedural floor bands + mullions,
    # in Object coords so the grid survives the Z rotation
    glass = bpy.data.materials.new("WH_Glass")
    glass.use_nodes = True
    nt = glass.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])

    def band_mask(axis_out, period, lo, hi):
        div = nt.nodes.new("ShaderNodeMath")
        div.operation = "DIVIDE"
        div.inputs[1].default_value = period
        nt.links.new(axis_out, div.inputs[0])
        frac = nt.nodes.new("ShaderNodeMath")
        frac.operation = "FRACT"
        nt.links.new(div.outputs["Value"], frac.inputs[0])
        gt = nt.nodes.new("ShaderNodeMath")
        gt.operation = "GREATER_THAN"
        gt.inputs[1].default_value = lo
        nt.links.new(frac.outputs["Value"], gt.inputs[0])
        lt = nt.nodes.new("ShaderNodeMath")
        lt.operation = "LESS_THAN"
        lt.inputs[1].default_value = hi
        nt.links.new(frac.outputs["Value"], lt.inputs[0])
        mul = nt.nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        nt.links.new(gt.outputs["Value"], mul.inputs[0])
        nt.links.new(lt.outputs["Value"], mul.inputs[1])
        return mul.outputs["Value"]

    floors = band_mask(sep.outputs["Z"], P["floor_h"], 0.10, 0.88)
    bays = band_mask(sep.outputs["X"], 3.4, 0.06, 0.94)
    wmask = nt.nodes.new("ShaderNodeMath")
    wmask.operation = "MULTIPLY"
    nt.links.new(floors, wmask.inputs[0])
    nt.links.new(bays, wmask.inputs[1])
    mixrgb = nt.nodes.new("ShaderNodeMix")
    mixrgb.data_type = "RGBA"
    nt.links.new(wmask.outputs["Value"], mixrgb.inputs["Factor"])
    mixrgb.inputs[6].default_value = (0.20, 0.19, 0.18, 1.0)  # mullion
    mixrgb.inputs[7].default_value = (0.01, 0.03, 0.06, 1.0)  # glass
    nt.links.new(mixrgb.outputs[2], bsdf.inputs["Base Color"])
    rough = nt.nodes.new("ShaderNodeMapRange")
    rough.inputs["From Min"].default_value = 0.0
    rough.inputs["From Max"].default_value = 1.0
    rough.inputs["To Min"].default_value = 0.6
    rough.inputs["To Max"].default_value = 0.12
    nt.links.new(wmask.outputs["Value"], rough.inputs["Value"])
    nt.links.new(rough.outputs["Result"], bsdf.inputs["Roughness"])
    metal = nt.nodes.new("ShaderNodeMath")
    metal.operation = "MULTIPLY"
    metal.inputs[1].default_value = 0.25
    nt.links.new(wmask.outputs["Value"], metal.inputs[0])
    nt.links.new(metal.outputs["Value"], bsdf.inputs["Metallic"])
    obj.data.materials.append(glass)

    # slot 1: mottled cast concrete
    conc = bpy.data.materials.new("WH_Concrete")
    conc.use_nodes = True
    nt2 = conc.node_tree
    bsdf2 = nt2.nodes["Principled BSDF"]
    bsdf2.inputs["Roughness"].default_value = 0.85
    tc2 = nt2.nodes.new("ShaderNodeTexCoord")
    noise = nt2.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 0.08
    noise.inputs["Detail"].default_value = 3.0
    nt2.links.new(tc2.outputs["Object"], noise.inputs["Vector"])
    ramp = nt2.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.50, 0.48, 0.46, 1.0)
    ramp.color_ramp.elements[1].color = (0.60, 0.58, 0.55, 1.0)
    nt2.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt2.links.new(ramp.outputs["Color"], bsdf2.inputs["Base Color"])
    obj.data.materials.append(conc)

    # slot 2: roof
    obj.data.materials.append(
        principled_material("WH_Roof", (0.18, 0.18, 0.19), roughness=0.9))


# --- buildings -----------------------------------------------------------------

def extrude_footprint(bm, ring, base_z, height, mat_index=0):
    """Add one extruded footprint into bmesh bm. Concave-safe."""
    coords = [Vector((x, y, 0.0)) for x, y in ring]
    tris = tessellate_polygon([coords])
    if not tris:
        return False
    top = base_z + height
    vb = [bm.verts.new((x, y, base_z)) for x, y in ring]
    vt = [bm.verts.new((x, y, top)) for x, y in ring]
    nv = len(ring)
    for i in range(nv):  # walls
        j = (i + 1) % nv
        try:
            bm.faces.new((vb[i], vb[j], vt[j], vt[i])).material_index = mat_index
        except ValueError:
            pass
    for a, b, c in tris:  # cap top and bottom
        try:
            bm.faces.new((vt[a], vt[b], vt[c])).material_index = mat_index
        except ValueError:
            pass
        try:
            bm.faces.new((vb[c], vb[b], vb[a])).material_index = mat_index
        except ValueError:
            pass
    return True


def building_mat_index(cx, cy, height, style):
    """Deterministic footprint-hash material slot."""
    a, b = int(round(cx * 8)), int(round(cy * 8))
    h = ((a * 73856093) ^ (b * 19349663)) & 0x7FFFFFFF
    if style == "schematic":
        return h % 3
    if height > 12.0:
        return 8 + h % 2  # window-band variants
    onsite = math.hypot(cx - facility.CAMPUS_CENTER[0],
                        cy - facility.CAMPUS_CENTER[1]) < facility.ONSITE_RADIUS
    return (4 + h % 4) if onsite else (h % 4)


def window_band_material(name, wall_color):
    """Horizontal dark-glass bands by world-space Z (for taller buildings)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.6
    geom = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geom.outputs["Position"], sep.inputs["Vector"])
    div = nt.nodes.new("ShaderNodeMath")
    div.operation = "DIVIDE"
    div.inputs[1].default_value = 3.2
    nt.links.new(sep.outputs["Z"], div.inputs[0])
    frac = nt.nodes.new("ShaderNodeMath")
    frac.operation = "FRACT"
    nt.links.new(div.outputs["Value"], frac.inputs[0])
    gt = nt.nodes.new("ShaderNodeMath")
    gt.operation = "GREATER_THAN"
    gt.inputs[1].default_value = 0.38
    nt.links.new(frac.outputs["Value"], gt.inputs[0])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    nt.links.new(gt.outputs["Value"], mix.inputs["Factor"])
    mix.inputs[6].default_value = (0.04, 0.05, 0.07, 1.0)  # window band
    mix.inputs[7].default_value = (*wall_color, 1.0)       # wall
    nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    mat.diffuse_color = (*wall_color, 1.0)
    return mat


def build_buildings(hm, style, cutaway=False, sculpt_wilson=True):
    with open(os.path.join(DATA, "buildings.geojson")) as f:
        feats = json.load(f)["features"]

    bm = bmesh.new()
    bm_wh = bmesh.new()
    skipped = 0
    wilson = None
    for f in feats:
        ring = f["geometry"]["coordinates"][0]
        # dedupe consecutive points
        ring = [p for i, p in enumerate(ring)
                if i == 0 or (abs(p[0] - ring[i - 1][0]) > 1e-6
                              or abs(p[1] - ring[i - 1][1]) > 1e-6)]
        if len(ring) < 3:
            skipped += 1
            continue
        cx = sum(p[0] for p in ring) / len(ring)
        cy = sum(p[1] for p in ring) / len(ring)
        if f["properties"].get("highlight"):
            wilson = (cx, cy)
            if sculpt_wilson:
                continue  # replaced by the sculpted model
        elif cutaway and in_cut(cx, cy):
            continue  # don't leave buildings floating over the pit
        base = 0.0 if style == "schematic" else hm.sample(cx, cy) - 3.0
        target = bm_wh if f["properties"].get("highlight") else bm
        mat_idx = 0 if target is bm_wh else building_mat_index(
            cx, cy, f["properties"]["height"], style)
        if not extrude_footprint(target, ring, base,
                                 f["properties"]["height"], mat_idx):
            skipped += 1

    mesh = bpy.data.meshes.new("BuildingsMesh")
    bm.to_mesh(mesh)
    bm.free()
    mesh.validate()
    obj = bpy.data.objects.new("Buildings", mesh)
    bpy.context.scene.collection.objects.link(obj)
    link_to(obj, "Buildings")
    if style == "schematic":
        for i, g in enumerate((0.52, 0.57, 0.62)):
            obj.data.materials.append(
                principled_material(f"BuildingMat{i}", (g, g, g + 0.02)))
    else:
        for i, c in enumerate(facility.BUILDING_PALETTE_OFFSITE):
            obj.data.materials.append(
                principled_material(f"BldgOff{i}", c, roughness=0.88))
        for i, c in enumerate(facility.BUILDING_PALETTE_ONSITE):
            obj.data.materials.append(
                principled_material(f"BldgOn{i}", c, roughness=0.75))
        obj.data.materials.append(
            window_band_material("BldgWinWarm", (0.45, 0.42, 0.38)))
        obj.data.materials.append(
            window_band_material("BldgWinCool", (0.40, 0.43, 0.47)))

    if sculpt_wilson:
        build_wilson_hall(hm, style)
    else:
        mesh_wh = bpy.data.meshes.new("WilsonHallMesh")
        bm_wh.to_mesh(mesh_wh)
        mesh_wh.validate()
        obj_wh = bpy.data.objects.new("WilsonHall", mesh_wh)
        bpy.context.scene.collection.objects.link(obj_wh)
        link_to(obj_wh, "Buildings")
        obj_wh.data.materials.append(
            emission_material("WilsonHallMat", (0.00, 0.45, 0.70),
                              strength=1.0, mix_principled=0.75))
    bm_wh.free()
    print(f"buildings: {len(feats)} footprints, {skipped} skipped")
    return wilson


# --- accelerator complex --------------------------------------------------------

def ring_curve(name, center, radius, depth, tube_r, color, strength):
    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "3D"
    sp = curve.splines.new("BEZIER")
    sp.bezier_points.add(3)  # 4 points -> circle
    kappa = 0.5523 * radius
    pts = [(radius, 0), (0, radius), (-radius, 0), (0, -radius)]
    hnd = [(0, kappa), (-kappa, 0), (0, -kappa), (kappa, 0)]
    for bp, (px, py), (hx, hy) in zip(sp.bezier_points, pts, hnd):
        bp.co = (center[0] + px, center[1] + py, depth)
        bp.handle_left = (center[0] + px - hx, center[1] + py - hy, depth)
        bp.handle_right = (center[0] + px + hx, center[1] + py + hy, depth)
        bp.handle_left_type = bp.handle_right_type = "FREE"
    sp.use_cyclic_u = True
    sp.resolution_u = 64
    curve.bevel_depth = tube_r
    curve.bevel_resolution = 4
    obj = bpy.data.objects.new(name, curve)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(
        emission_material(name + "Mat", color, strength=strength,
                          mix_principled=0.3))
    link_to(obj, "Accelerators")
    return obj


def segment_curve(name, start, end, tube_r, color, strength):
    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "3D"
    sp = curve.splines.new("POLY")
    sp.points.add(1)
    sp.points[0].co = (*start, 1.0)
    sp.points[1].co = (*end, 1.0)
    curve.bevel_depth = tube_r
    curve.bevel_resolution = 3
    curve.use_fill_caps = True
    obj = bpy.data.objects.new(name, curve)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(
        emission_material(name + "Mat", color, strength=strength,
                          mix_principled=0.3))
    link_to(obj, "Accelerators")
    return obj


def azimuth_point(ring_cfg, azimuth_deg):
    r = facility.ring_radius(ring_cfg)
    a = math.radians(azimuth_deg)
    cx, cy = ring_cfg["center"]
    return cx + r * math.cos(a), cy + r * math.sin(a)


def build_accelerators(hm, style):
    rings = {r["name"]: r for r in facility.RINGS}
    strength = 3.5 if style == "schematic" else 1.0
    for r in facility.RINGS:
        radius = facility.ring_radius(r) + r.get("radius_offset", 0.0)
        ring_curve("Ring_" + r["name"], r["center"], radius,
                   r["depth"], r["tube_radius"], r["color"], strength)
    for seg in facility.LINACS:
        segment_curve("Seg_" + seg["name"], seg["start"], seg["end"],
                      seg["tube_radius"], seg["color"], strength)

    # target hall
    th = facility.TARGET_HALL
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=th["center"])
    obj = bpy.context.object
    obj.scale = th["size"]
    obj.name = "TargetHall"
    obj.data.materials.append(
        emission_material("TargetHallMat", th["color"], strength=3.0,
                          mix_principled=0.2))
    link_to(obj, "Infrastructure")

    # IP caverns + detectors on the collider ring
    col = rings["Collider"]
    for ip in facility.IP_CAVERNS:
        x, y = azimuth_point(col, ip["azimuth_deg"])
        loc = (x, y, col["depth"])
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)
        cav = bpy.context.object
        cav.scale = facility.IP_CAVERN_SIZE
        cav.name = "Cavern_" + ip["name"]
        cav.data.materials.append(
            emission_material(ip["name"] + "CavMat", (0.35, 0.35, 0.40),
                              strength=0.8, mix_principled=0.5))
        link_to(cav, "Infrastructure")
        bpy.ops.mesh.primitive_cylinder_add(radius=8.0, depth=16.0, location=loc,
                                            rotation=(0, math.pi / 2, 0))
        det = bpy.context.object
        det.name = "Detector_" + ip["name"]
        det.data.materials.append(
            emission_material(ip["name"] + "DetMat", facility.IP_COLOR,
                              strength=5.0))
        link_to(det, "Infrastructure")

    # access shafts
    for ring_name, az in facility.SHAFTS:
        r = rings[ring_name]
        x, y = azimuth_point(r, az)
        top = 0.0 if style == "schematic" else hm.sample(x, y)
        depth = r["depth"]
        h = top - depth
        bpy.ops.mesh.primitive_cylinder_add(radius=facility.SHAFT_RADIUS,
                                            depth=h,
                                            location=(x, y, top - h / 2))
        obj = bpy.context.object
        obj.name = f"Shaft_{ring_name}_{int(az)}"
        obj.data.materials.append(
            emission_material(obj.name + "Mat", (0.9, 0.9, 0.9), strength=1.5,
                              mix_principled=0.5))
        link_to(obj, "Infrastructure")


# --- annotations -----------------------------------------------------------------

def text_object(name, text, location, size, color, rotation=(0, 0, 0),
                align="CENTER"):
    curve = bpy.data.curves.new(name, type="FONT")
    curve.body = text
    curve.size = size
    curve.extrude = size * 0.02
    curve.align_x = align
    obj = bpy.data.objects.new(name, curve)
    obj.location = location
    obj.rotation_euler = rotation
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(
        emission_material(name + "Mat", color, strength=2.5))
    obj.visible_shadow = False  # no giant text shadows on the terrain
    link_to(obj, "Annotations")
    return obj


def build_annotations(style):
    cx, cy = facility.CAMPUS_CENTER
    tcol = (0.9, 0.9, 0.95) if style == "schematic" else (0.05, 0.05, 0.07)

    text_object("Title", "Fermilab 10 TeV Muon Collider (site-filler concept)",
                (cx, cy + 3500.0, 80.0), 240.0, tcol)
    text_object("Subtitle",
                "arXiv:2503.23695 - schematic; rings drawn as circles",
                (cx, cy + 3180.0, 80.0), 115.0, tcol)

    # legend: swatch + label rows, below the site's south edge
    entries = [(r["label"], r["color"]) for r in facility.RINGS]
    entries += [(s["label"], s["color"]) for s in facility.LINACS]
    entries += [(facility.TARGET_HALL["label"], facility.TARGET_HALL["color"]),
                ("IP detector halls (x2)", facility.IP_COLOR),
                ("Wilson Hall", (0.00, 0.45, 0.70))]
    x0, y0 = cx - 6300.0, cy + 1300.0
    row = 240.0
    for i, (label, color) in enumerate(entries):
        y = y0 - i * row
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x0, y + 50.0, 40.0))
        sw = bpy.context.object
        sw.scale = (130.0, 130.0, 20.0)
        sw.name = f"LegendSwatch_{i}"
        sw.data.materials.append(
            emission_material(f"LegendSwatchMat_{i}", color, strength=1.8))
        sw.visible_shadow = False
        link_to(sw, "Annotations")
        text_object(f"LegendText_{i}", label, (x0 + 220.0, y, 40.0), 140.0,
                    tcol, align="LEFT")


# --- cameras & lights --------------------------------------------------------------

def add_camera(name, location, look_at, ortho_scale=None, lens=35.0):
    cam_data = bpy.data.cameras.new(name)
    if ortho_scale:
        cam_data.type = "ORTHO"
        cam_data.ortho_scale = ortho_scale
    else:
        cam_data.lens = lens
    cam_data.clip_end = 100000.0
    cam = bpy.data.objects.new(name, cam_data)
    cam.location = location
    direction = Vector(look_at) - Vector(location)
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.collection.objects.link(cam)
    link_to(cam, "Cameras")
    return cam


def build_cameras(style, wilson_xy, hm):
    cx, cy = facility.CAMPUS_CENTER
    cams = []
    cams.append(add_camera("Cam_Aerial", (cx - 3800, cy - 6800, 5200),
                           (cx, cy, -350), lens=38.0))
    # aim at the exposed cut quadrant (southeast)
    cams.append(add_camera("Cam_Cutaway",
                           (cx + 5200, cy - 5600, 2600),
                           (cx + 900, cy - 900, -180), lens=40.0))
    if style == "schematic":
        cams.append(add_camera("Cam_Top", (cx, cy, 9500), (cx, cy, 0),
                               ortho_scale=13200))
    else:
        wx, wy = wilson_xy or (0.0, 0.0)
        wz = hm.sample(wx, wy)
        # from the ENE along the long axis: shows the iconic end-on profile
        # (two curved pylons + atrium slot) plus the sunlit SE glass face
        cams.append(add_camera("Cam_WilsonHall",
                               (wx + 620, wy + 190, wz + 200),
                               (wx, wy, wz + 42), lens=45.0))
    bpy.context.scene.camera = cams[0]
    bpy.context.scene["render_cameras"] = [c.name for c in cams]


def build_lights(style):
    sun_data = bpy.data.lights.new("Sun", type="SUN")
    sun_data.energy = 3.0
    sun_data.angle = math.radians(1.0)
    sun = bpy.data.objects.new("Sun", sun_data)
    # from the southwest, ~33 deg elevation for stronger modeling shadows
    sun.rotation_euler = (math.radians(57.0), 0.0, math.radians(45.0))
    bpy.context.scene.collection.objects.link(sun)
    link_to(sun, "Lights")

    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    if style == "schematic":
        bg.inputs["Color"].default_value = (0.92, 0.93, 0.95, 1.0)
        bg.inputs["Strength"].default_value = 0.8
    else:
        sky = world.node_tree.nodes.new("ShaderNodeTexSky")
        sky.sun_elevation = math.radians(33.0)
        sky.sun_rotation = math.radians(135.0)
        sky.sun_intensity = 0.3
        world.node_tree.links.new(sky.outputs["Color"], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = 0.6


# --- main ---------------------------------------------------------------------------

def setup_render(fast):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 32 if fast else 128
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = "OPENIMAGEDENOISE"
    sc.render.resolution_x = 640 if fast else 1920
    sc.render.resolution_y = 360 if fast else 1080
    sc.render.film_transparent = False
    style = bpy.context.scene.get("style", "")
    sc.view_settings.view_transform = (
        "Standard" if style == "schematic" else "Filmic")
    sc.render.image_settings.file_format = "PNG"


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", choices=["schematic", "realistic"],
                    required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--no-cutaway", action="store_true")
    ap.add_argument("--no-wilson-model", action="store_true")
    args = ap.parse_args(argv)

    # empty scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    hm = Heightmap(os.path.join(DATA, "heightmap.npz"))
    cutaway = (args.style == "realistic") and not args.no_cutaway

    build_terrain(hm, args.style, cutaway)
    build_boundary(hm, args.style)
    wilson_xy = build_buildings(hm, args.style, cutaway,
                                sculpt_wilson=not args.no_wilson_model)
    if args.style == "realistic":
        build_water(hm, cutaway)
    build_accelerators(hm, args.style)
    build_annotations(args.style)
    build_cameras(args.style, wilson_xy, hm)
    build_lights(args.style)
    bpy.context.scene["style"] = args.style
    setup_render(args.fast)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.out),
                                compress=True)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
