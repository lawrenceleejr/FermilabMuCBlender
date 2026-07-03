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
    curve.bevel_depth = facility.BOUNDARY_RADIUS
    curve.bevel_resolution = 3
    obj = bpy.data.objects.new("SiteBoundary", curve)
    bpy.context.scene.collection.objects.link(obj)
    # bright, identical in both styles: the campus outline is a key datum
    obj.data.materials.append(
        emission_material("BoundaryMat", facility.BOUNDARY_COLOR,
                          strength=facility.BOUNDARY_STRENGTH))
    obj.visible_shadow = False
    # lives with the annotations: bright and legible in the map-like views,
    # hidden in the ground-level hero shot where it would cross the horizon
    link_to(obj, "Annotations")


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
    """Sculpted model matched to reference photos: vertical outer faces
    with horizontal window strips; atrium-side inner faces sweeping from a
    narrow top slot to a wide wishbone stance; flat concrete slab ends;
    vertically-mulled atrium glazing; rooftop parapet ears; splayed
    abutment walls. Slots: 0 atrium glass, 1 board-form concrete (inner
    faces/ears/abutments), 2 roof/canopy, 3 outer window bands, 4 end
    panel concrete."""
    P = facility.WILSON_HALL_MODEL
    H = P["floors"] * P["floor_h"]
    L = P["length"]

    def profiles(z):
        t = z / H
        if t >= P["t_waist"]:
            # zero slope at the roof: the outer walls rise parallel at the
            # top and only curve inward approaching the waist
            u = (1.0 - t) / (1.0 - P["t_waist"])
            outer = (P["w_top_half"]
                     - (P["w_top_half"] - P["w_waist_half"])
                     * u ** P["upper_exp"])
        else:
            u = (P["t_waist"] - t) / P["flare_span"]
            outer = P["w_waist_half"] + P["flare"] * u ** P["flare_exp"]
        if t >= P["t_slot"]:
            gap = P["slot_half"]
        else:
            u = (P["t_slot"] - t) / P["t_slot"]
            gap = (P["slot_half"]
                   + (P["gap_base_half"] - P["slot_half"]) * u ** P["gap_exp"])
        return outer, gap

    bm = bmesh.new()
    slot_map = {}
    nlev = 4 * P["floors"]  # quarter-floor loft: smooth high-curvature sweep
    levels = [k * P["floor_h"] / 4.0 for k in range(nlev + 1)]
    for sign in (1.0, -1.0):
        rings = []
        for z in levels:
            outer, gap = profiles(z)
            y0, y1 = sign * gap, sign * outer
            rings.append([bm.verts.new(v) for v in
                          ((-L / 2, y0, z), (L / 2, y0, z),
                           (L / 2, y1, z), (-L / 2, y1, z))])
        for k in range(nlev):
            a, b = rings[k], rings[k + 1]
            for e in range(4):
                f = bm.faces.new((a[e], a[(e + 1) % 4],
                                  b[(e + 1) % 4], b[e]))
                # e==0: curved atrium-side face -> concrete (1)
                # e==2: vertical outer face -> window strips (3)
                # e==1,3: flat slab ends -> panel concrete (4)
                slot_map[f] = {0: 1, 2: 3}.get(e, 4)
                f.smooth = e in (0, 2)
        slot_map[bm.faces.new(rings[0])] = 1
        slot_map[bm.faces.new(list(reversed(rings[-1])))] = 2

        # distinctive rooftop blocks: one per pylon at each end, flanking
        # the central slot (per reference photos)
        bl, bw, bh = P["roof_block"]
        _, gap_top = profiles(H)
        for ex in (L / 2 - bl / 2 - 0.5, -L / 2 + bl / 2 + 0.5):
            _add_box(bm, (ex, sign * (gap_top + bw / 2), H + bh / 2),
                     (bl, bw, bh), slot_map, 1)

    # atrium glazing closing both ends (wide base sweeping into the slot);
    # it tops out below the roof - the notch above stays open to the sky
    z_glass_top = P["t_glass_top"] * H
    for k in range(nlev):
        z0, z1 = levels[k], levels[k + 1]
        if z0 >= z_glass_top:
            break
        z1 = min(z1, z_glass_top)
        _, g0 = profiles(z0)
        _, g1 = profiles(z1)
        for sx in (L / 2, -L / 2):
            va = bm.verts.new((sx, -g0, z0))
            vb = bm.verts.new((sx, g0, z0))
            vc = bm.verts.new((sx, g1, z1))
            vd = bm.verts.new((sx, -g1, z1))
            f = bm.faces.new((va, vb, vc, vd))
            slot_map[f] = 0
            f.smooth = True
    # concrete transom beam capping the glass at each end
    _, g_cap = profiles(z_glass_top)
    for sx in (L / 2 - 1.2, -L / 2 + 1.2):
        _add_box(bm, (sx, 0.0, z_glass_top + 0.8),
                 (2.4, 2 * g_cap, 1.6), slot_map, 1)

    # entrance canopies + splayed abutment walls at grade
    _, gap_base = profiles(0.0)
    for sx in (1.0, -1.0):
        _add_box(bm, (sx * (L / 2 + 4.0), 0.0, 4.6),
                 (9.0, 2 * gap_base * 0.8, 0.9), slot_map, 2)
        for sy in (1.0, -1.0):
            _add_box(bm, (sx * (L / 2 + 14.0), sy * (gap_base + 4.0), 1.8),
                     (26.0, 3.0, 3.6), slot_map, 1)

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
        for _ in range(5):
            obj.data.materials.append(m)
        return

    def band_nodes(nt, axis_out, period, lo, hi):
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

    def masked_material(name, mask_builder, wall_rgb, dark_rgb,
                        wall_rough=0.85, dark_rough=0.15, metallic=0.0):
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nt = mat.node_tree
        bsdf = nt.nodes["Principled BSDF"]
        tc = nt.nodes.new("ShaderNodeTexCoord")
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])
        mask = mask_builder(nt, sep)
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        nt.links.new(mask, mix.inputs["Factor"])
        mix.inputs[6].default_value = (*wall_rgb, 1.0)
        mix.inputs[7].default_value = (*dark_rgb, 1.0)
        nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
        rough = nt.nodes.new("ShaderNodeMapRange")
        rough.inputs["To Min"].default_value = wall_rough
        rough.inputs["To Max"].default_value = dark_rough
        nt.links.new(mask, rough.inputs["Value"])
        nt.links.new(rough.outputs["Result"], bsdf.inputs["Roughness"])
        if metallic:
            met = nt.nodes.new("ShaderNodeMath")
            met.operation = "MULTIPLY"
            met.inputs[1].default_value = metallic
            nt.links.new(mask, met.inputs[0])
            nt.links.new(met.outputs["Value"], bsdf.inputs["Metallic"])
        return mat

    fh = P["floor_h"]

    # slot 0: atrium glazing - strong vertical mullions (bays across Y),
    # faint floor transoms, dark green-blue glass
    def glass_mask(nt, sep):
        bays = band_nodes(nt, sep.outputs["Y"], 1.5, 0.06, 0.94)
        floors = band_nodes(nt, sep.outputs["Z"], fh, 0.04, 0.97)
        mul = nt.nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        nt.links.new(bays, mul.inputs[0])
        nt.links.new(floors, mul.inputs[1])
        return mul.outputs["Value"]

    obj.data.materials.append(masked_material(
        "WH_AtriumGlass", glass_mask,
        (0.05, 0.05, 0.05), (0.004, 0.009, 0.012),
        wall_rough=0.55, dark_rough=0.10, metallic=0.15))

    # slot 1: board-formed concrete (curved inner faces, ears, abutments)
    conc = bpy.data.materials.new("WH_Concrete")
    conc.use_nodes = True
    nt2 = conc.node_tree
    bsdf2 = nt2.nodes["Principled BSDF"]
    bsdf2.inputs["Roughness"].default_value = 0.88
    tc2 = nt2.nodes.new("ShaderNodeTexCoord")
    noise = nt2.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 0.35
    noise.inputs["Detail"].default_value = 4.0
    nt2.links.new(tc2.outputs["Object"], noise.inputs["Vector"])
    ramp = nt2.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.52, 0.50, 0.47, 1.0)
    ramp.color_ramp.elements[1].color = (0.63, 0.61, 0.57, 1.0)
    nt2.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt2.links.new(ramp.outputs["Color"], bsdf2.inputs["Base Color"])
    obj.data.materials.append(conc)

    # slot 2: roof / canopy
    obj.data.materials.append(
        principled_material("WH_Roof", (0.20, 0.20, 0.21), roughness=0.9))

    # slot 3: vertical outer faces - 16 horizontal window strips between
    # concrete spandrels
    def strip_mask(nt, sep):
        return band_nodes(nt, sep.outputs["Z"], fh, 0.34, 0.74)

    obj.data.materials.append(masked_material(
        "WH_WindowStrips", strip_mask,
        (0.58, 0.56, 0.52), (0.02, 0.04, 0.05),
        wall_rough=0.85, dark_rough=0.12, metallic=0.2))

    # slot 4: flat slab ends - precast panels, thin joints every two floors
    def joint_mask(nt, sep):
        return band_nodes(nt, sep.outputs["Z"], 2 * fh, 0.0, 0.018)

    obj.data.materials.append(masked_material(
        "WH_EndPanels", joint_mask,
        (0.60, 0.58, 0.54), (0.38, 0.36, 0.33),
        wall_rough=0.88, dark_rough=0.88))


def build_flags(hm, style):
    """Single row of international flags rippling in the wind, crossing
    in front of the main (NE) entrance as in the reference photos."""
    if style != "realistic":
        return
    P = facility.WILSON_HALL_MODEL
    F = facility.FLAG_ROW
    ang = math.radians(90.0 - P["rotation_deg"])
    ca, sa = math.cos(ang), math.sin(ang)

    def to_world(lx, ly, lz):
        return (lx * ca - ly * sa, lx * sa + ly * ca, lz)

    bm = bmesh.new()
    slot_map = {}
    nu, nv = 10, 4  # flag cloth grid
    fl, fh_ = F["flag_l"], F["flag_h"]
    idx = 0
    ly = -F["y_span"]
    while ly <= F["y_span"] + 0.01:
        wxp, wyp, _ = to_world(F["x"], ly, 0.0)
        gz = hm.sample(wxp, wyp)
        _add_box(bm, (wxp, wyp, gz + F["pole_h"] / 2),
                 (0.18, 0.18, F["pole_h"]), slot_map, 0)

        # flag cloth: ripples along its length, blowing along the row
        phase = idx * 1.7
        color_slot = 1 + (idx % len(facility.FLAG_COLORS))
        grid = []
        for i in range(nu):
            u = i / (nu - 1)
            ripple = 0.30 * math.sin(u * 5.0 + phase) * u
            droop = 0.22 * u * u
            col = []
            for j in range(nv):
                v = j / (nv - 1)
                lz = gz + F["pole_h"] - 0.25 - v * fh_ - droop
                col.append(bm.verts.new(
                    to_world(F["x"] + ripple, ly + 0.12 + u * fl, lz)))
            grid.append(col)
        for i in range(nu - 1):
            for j in range(nv - 1):
                f = bm.faces.new((grid[i][j], grid[i + 1][j],
                                  grid[i + 1][j + 1], grid[i][j + 1]))
                slot_map[f] = color_slot
                f.smooth = True
        idx += 1
        ly += F["spacing"]

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mat_idx = [slot_map.get(f, 0) for f in bm.faces]
    mesh = bpy.data.meshes.new("FlagsMesh")
    bm.to_mesh(mesh)
    bm.free()
    mesh.validate()
    mesh.polygons.foreach_set("material_index", mat_idx)
    obj = bpy.data.objects.new("Flags", mesh)
    bpy.context.scene.collection.objects.link(obj)
    link_to(obj, "Buildings")
    obj.data.materials.append(
        principled_material("FlagPole", (0.85, 0.86, 0.88), roughness=0.4))
    for i, c in enumerate(facility.FLAG_COLORS):
        mat = principled_material(f"Flag{i}", c, roughness=0.85)
        mat.node_tree.nodes["Principled BSDF"].inputs[
            "Sheen Weight"].default_value = 0.4
        obj.data.materials.append(mat)


def helen_edwards_material():
    """Modern glass lab: vertical glazing bays with metal fins."""
    mat = bpy.data.materials.new("HelenEdwardsGlass")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])
    add = nt.nodes.new("ShaderNodeMath")
    add.operation = "ADD"
    nt.links.new(sep.outputs["X"], add.inputs[0])
    nt.links.new(sep.outputs["Y"], add.inputs[1])
    div = nt.nodes.new("ShaderNodeMath")
    div.operation = "DIVIDE"
    div.inputs[1].default_value = 2.8
    nt.links.new(add.outputs["Value"], div.inputs[0])
    frac = nt.nodes.new("ShaderNodeMath")
    frac.operation = "FRACT"
    nt.links.new(div.outputs["Value"], frac.inputs[0])
    gt = nt.nodes.new("ShaderNodeMath")
    gt.operation = "GREATER_THAN"
    gt.inputs[1].default_value = 0.12
    nt.links.new(frac.outputs["Value"], gt.inputs[0])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    nt.links.new(gt.outputs["Value"], mix.inputs["Factor"])
    mix.inputs[6].default_value = (0.55, 0.57, 0.58, 1.0)  # metal fin
    mix.inputs[7].default_value = (0.03, 0.07, 0.10, 1.0)  # glass
    nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    rough = nt.nodes.new("ShaderNodeMapRange")
    rough.inputs["To Min"].default_value = 0.45
    rough.inputs["To Max"].default_value = 0.10
    nt.links.new(gt.outputs["Value"], rough.inputs["Value"])
    nt.links.new(rough.outputs["Result"], bsdf.inputs["Roughness"])
    bsdf.inputs["Metallic"].default_value = 0.4
    return mat


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
    bm_hel = bmesh.new()
    skipped = 0
    wilson = None
    hx, hy = facility.HELEN_EDWARDS_CENTROID
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
        if math.hypot(cx - hx, cy - hy) < 40.0:
            # Helen Edwards Laboratory (former IERC): modern glass lab
            extrude_footprint(bm_hel, ring, base,
                              facility.HELEN_EDWARDS_HEIGHT, 0)
            continue
        target = bm_wh if f["properties"].get("highlight") else bm
        mat_idx = 0 if target is bm_wh else building_mat_index(
            cx, cy, f["properties"]["height"], style)
        if not extrude_footprint(target, ring, base,
                                 f["properties"]["height"], mat_idx):
            skipped += 1

    mesh_hel = bpy.data.meshes.new("HelenEdwardsMesh")
    bm_hel.to_mesh(mesh_hel)
    bm_hel.free()
    mesh_hel.validate()
    obj_hel = bpy.data.objects.new("HelenEdwardsLab", mesh_hel)
    bpy.context.scene.collection.objects.link(obj_hel)
    link_to(obj_hel, "Buildings")
    if style == "schematic":
        obj_hel.data.materials.append(
            principled_material("HelenEdwardsMat", (0.35, 0.55, 0.65)))
    else:
        obj_hel.data.materials.append(helen_edwards_material())

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
    strength = 3.5 if style == "schematic" else 2.5  # dusk glow in realistic
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


def build_annotations(style, hm):
    cx, cy = facility.CAMPUS_CENTER
    tcol = (0.9, 0.9, 0.95) if style == "schematic" else (0.05, 0.05, 0.07)

    def ground(x, y):
        # annotations lie just above the surface: readable from above,
        # edge-on (invisible) from ground-level and tour cameras
        return 3.0 if style == "schematic" else hm.sample(x, y) + 3.0

    text_object("Title", "Fermilab 10 TeV Muon Collider (site-filler concept)",
                (cx, cy + 3500.0, ground(cx, cy + 3500.0)), 240.0, tcol)
    text_object("Subtitle",
                "arXiv:2503.23695 - schematic; rings drawn as circles",
                (cx, cy + 3180.0, ground(cx, cy + 3180.0)), 115.0, tcol)

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
        gz = ground(x0, y)
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x0, y + 50.0, gz))
        sw = bpy.context.object
        sw.scale = (130.0, 130.0, 4.0)
        sw.name = f"LegendSwatch_{i}"
        sw.data.materials.append(
            emission_material(f"LegendSwatchMat_{i}", color, strength=1.8))
        sw.visible_shadow = False
        link_to(sw, "Annotations")
        text_object(f"LegendText_{i}", label, (x0 + 220.0, y, gz), 140.0,
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
    # lower, layered oblique: pit + rings mid-frame, campus + horizon beyond
    cams.append(add_camera("Cam_Aerial", (cx - 2400, cy - 7200, 3000),
                           (cx + 300, cy - 200, -350), lens=44.0))
    # low over the pit rim, wide: the cut wall drops away, rings sweep out
    cams.append(add_camera("Cam_Cutaway",
                           (cx + 4300, cy - 4300, 1200),
                           (cx + 700, cy - 700, -220), lens=28.0))
    if style == "schematic":
        cams.append(add_camera("Cam_Top", (cx, cy, 9500), (cx, cy, 0),
                               ortho_scale=13200))
    else:
        wx, wy = wilson_xy or (0.0, 0.0)
        wz = hm.sample(wx, wy)
        # near end-on from the SSW (sunlit side): the iconic wishbone pylons
        # + atrium glass arch, slight 3/4 so the striped long face reads
        # from the NE: the entrance approach with the flag rows and the
        # Helen Edwards Laboratory in frame
        cams.append(add_camera("Cam_WilsonHall",
                               (wx + 336, wy + 284, wz + 20),
                               (wx, wy, wz + 40), lens=40.0))
    bpy.context.scene.camera = cams[0]
    bpy.context.scene["render_cameras"] = [c.name for c in cams]


def build_tour_camera(style, hm):
    """Keyframed camera that tours the complex (embedded in the .blend;
    not rendered in CI). Render with: blender -b <file> -a"""
    sc = bpy.context.scene
    sc.frame_start = 1
    sc.frame_end = 720
    sc.render.fps = 24

    target = bpy.data.objects.new("TourTarget", None)
    target.empty_display_size = 50.0
    bpy.context.scene.collection.objects.link(target)
    link_to(target, "Cameras")

    cam_data = bpy.data.cameras.new("Cam_Tour")
    cam_data.lens = 32.0
    cam_data.clip_end = 100000.0
    cam = bpy.data.objects.new("Cam_Tour", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    link_to(cam, "Cameras")
    con = cam.constraints.new(type="TRACK_TO")
    con.target = target
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"

    cx, cy = facility.CAMPUS_CENTER
    wz = 0.0 if style == "schematic" else hm.sample(0.0, 0.0)
    # (frame, camera location, target location)
    waypoints = [
        (1,   (cx - 3000, cy - 8500, 4800), (cx, cy, -200)),        # wide reveal
        (140, (-1600, -2600, 1100),         (0.0, 0.0, wz + 50)),   # dive toward Wilson Hall
        (260, (620, 320, wz + 150),         (0.0, 0.0, wz + 45)),   # hero orbit-in
        (360, (900, -900, wz + 420),        (1500, -300, -50)),     # rise, turn to the pit
        (500, (cx + 2900, cy - 2900, 750),  (cx + 500, cy - 500, -200)),  # over the cut, rings below
        (640, (cx + 700, cy - 5800, 2600),  (cx, cy, -250)),        # sweep along the south
        (720, (cx - 3000, cy - 8500, 4800), (cx, cy, -200)),        # settle back wide
    ]
    for frame, loc, tgt in waypoints:
        cam.location = loc
        cam.keyframe_insert(data_path="location", frame=frame)
        target.location = tgt
        target.keyframe_insert(data_path="location", frame=frame)
    for obj in (cam, target):
        for fc in obj.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "BEZIER"
                kp.easing = "EASE_IN_OUT"


def build_lights(style):
    sun_data = bpy.data.lights.new("Sun", type="SUN")
    sun = bpy.data.objects.new("Sun", sun_data)
    bpy.context.scene.collection.objects.link(sun)
    link_to(sun, "Lights")

    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    if style == "schematic":
        # dark studio backdrop: glowing rings + white buildings pop
        sun_data.energy = 3.0
        sun_data.angle = math.radians(1.0)
        sun.rotation_euler = (math.radians(57.0), 0.0, math.radians(45.0))
        bg.inputs["Color"].default_value = (0.015, 0.02, 0.035, 1.0)
        bg.inputs["Strength"].default_value = 1.0
    else:
        # golden hour: low warm sun from the WSW, warm horizon sky; the pit
        # falls into shadow so the emissive machines glow against dusk
        sun_data.energy = 6.0
        sun_data.angle = math.radians(0.6)
        sun_data.color = (1.0, 0.60, 0.34)
        # low SE sun, ~8 deg elevation: golden-hour rake on the entrance
        # end, the flags, and the pit walls
        sun.rotation_euler = (math.radians(82.0), 0.0, math.radians(45.0))
        sky = world.node_tree.nodes.new("ShaderNodeTexSky")
        sky.sun_elevation = math.radians(8.0)
        sky.sun_rotation = math.radians(20.0)
        sky.sun_intensity = 0.3
        sky.dust_density = 1.5
        world.node_tree.links.new(sky.outputs["Color"], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = 0.55
    # mist for aerial perspective (read by the compositor)
    world.mist_settings.start = 3000.0
    world.mist_settings.depth = 14000.0
    world.mist_settings.falloff = "QUADRATIC"


def setup_compositor(style):
    """Magazine grade: mist-based aerial perspective (realistic), fog-glow
    on the emissive machines, and a soft vignette."""
    sc = bpy.context.scene
    sc.render.use_compositing = True
    sc.use_nodes = True
    vl = bpy.context.view_layer
    vl.use_pass_mist = True
    nt = sc.node_tree
    nt.nodes.clear()
    rl = nt.nodes.new("CompositorNodeRLayers")
    img_out = rl.outputs["Image"]

    if style == "realistic":
        haze = nt.nodes.new("CompositorNodeMixRGB")
        haze.blend_type = "MIX"
        haze.inputs[2].default_value = (0.72, 0.52, 0.34, 1.0)  # golden haze
        scale_mist = nt.nodes.new("CompositorNodeMath")
        scale_mist.operation = "MULTIPLY"
        scale_mist.inputs[1].default_value = 0.26
        nt.links.new(rl.outputs["Mist"], scale_mist.inputs[0])
        nt.links.new(scale_mist.outputs["Value"], haze.inputs["Fac"])
        nt.links.new(img_out, haze.inputs[1])
        img_out = haze.outputs["Image"]

    glare = nt.nodes.new("CompositorNodeGlare")
    glare.glare_type = "FOG_GLOW"
    glare.quality = "HIGH"
    # options moved between properties and sockets across 4.x - set both ways
    for attr, val in (("threshold", 1.0), ("size", 8)):
        try:
            setattr(glare, attr, val)
        except (AttributeError, TypeError):
            pass
    for sock, val in (("Threshold", 1.0), ("Size", 0.45), ("Strength", 0.08)):
        if sock in glare.inputs:
            try:
                glare.inputs[sock].default_value = val
            except (AttributeError, TypeError):
                pass
    nt.links.new(img_out, glare.inputs["Image"])
    img_out = glare.outputs["Image"]

    comp = nt.nodes.new("CompositorNodeComposite")
    nt.links.new(img_out, comp.inputs["Image"])


# --- main ---------------------------------------------------------------------------

def setup_render(fast):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 32 if fast else 128
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = "OPENIMAGEDENOISE"
    sc.render.resolution_x = 640 if fast else 2560
    sc.render.resolution_y = 360 if fast else 1440
    sc.render.film_transparent = False
    style = bpy.context.scene.get("style", "")
    if style == "schematic":
        sc.view_settings.view_transform = "Standard"
    else:
        try:
            sc.view_settings.view_transform = "AgX"
            sc.view_settings.look = "AgX - Punchy"
        except TypeError:
            sc.view_settings.view_transform = "Filmic"
            sc.view_settings.look = "High Contrast"
        sc.view_settings.exposure = -0.35  # darker, moodier frames
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
    build_flags(hm, args.style)
    build_accelerators(hm, args.style)
    build_annotations(args.style, hm)
    build_cameras(args.style, wilson_xy, hm)
    build_tour_camera(args.style, hm)
    build_lights(args.style)
    bpy.context.scene["style"] = args.style
    setup_render(args.fast)
    setup_compositor(args.style)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.out),
                                compress=True)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
