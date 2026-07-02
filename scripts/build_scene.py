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
    # surface tint by elevation
    ramp_surf = nt.nodes.new("ShaderNodeValToRGB")
    ramp_surf.color_ramp.elements[0].position = 0.0
    ramp_surf.color_ramp.elements[0].color = (0.06, 0.18, 0.04, 1.0)  # low: green
    ramp_surf.color_ramp.elements[1].position = 1.0
    ramp_surf.color_ramp.elements[1].color = (0.28, 0.24, 0.11, 1.0)  # high: brown
    map_surf = nt.nodes.new("ShaderNodeMapRange")
    map_surf.inputs["From Min"].default_value = -20.0
    map_surf.inputs["From Max"].default_value = 25.0
    nt.links.new(sep.outputs["Z"], map_surf.inputs["Value"])
    nt.links.new(map_surf.outputs["Result"], ramp_surf.inputs["Fac"])
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
    # blend surface vs underground by z threshold
    thresh = nt.nodes.new("ShaderNodeMath")
    thresh.operation = "GREATER_THAN"
    thresh.inputs[1].default_value = -2.0
    nt.links.new(sep.outputs["Z"], thresh.inputs[0])
    mixc = nt.nodes.new("ShaderNodeMix")
    mixc.data_type = "RGBA"
    nt.links.new(thresh.outputs["Value"], mixc.inputs["Factor"])
    nt.links.new(ramp_geo.outputs["Color"], mixc.inputs[6])   # A: underground
    nt.links.new(ramp_surf.outputs["Color"], mixc.inputs[7])  # B: surface
    # faint emission so the pit interior isn't pitch black in shadow
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = 0.10
    nt.links.new(mixc.outputs[2], em.inputs["Color"])
    add = nt.nodes.new("ShaderNodeAddShader")
    out_node = nt.nodes["Material Output"]
    nt.links.new(bsdf.outputs["BSDF"], add.inputs[0])
    nt.links.new(em.outputs["Emission"], add.inputs[1])
    nt.links.new(add.outputs["Shader"], out_node.inputs["Surface"])
    nt.links.new(mixc.outputs[2], bsdf.inputs["Base Color"])
    obj.data.materials.append(mat)

    # large backdrop plane so the scene doesn't float on a black void
    bpy.ops.mesh.primitive_plane_add(size=60000.0,
                                     location=(facility.CAMPUS_CENTER[0],
                                               facility.CAMPUS_CENTER[1],
                                               facility.TERRAIN_SOLID_BOTTOM - 5.0))
    bd = bpy.context.object
    bd.name = "Backdrop"
    bd.data.materials.append(
        principled_material("BackdropMat", (0.10, 0.16, 0.07), roughness=1.0))
    link_to(bd, "Terrain")
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


# --- buildings -----------------------------------------------------------------

def extrude_footprint(bm, ring, base_z, height):
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
            bm.faces.new((vb[i], vb[j], vt[j], vt[i]))
        except ValueError:
            pass
    for a, b, c in tris:  # cap top and bottom
        try:
            bm.faces.new((vt[a], vt[b], vt[c]))
        except ValueError:
            pass
        try:
            bm.faces.new((vb[c], vb[b], vb[a]))
        except ValueError:
            pass
    return True


def build_buildings(hm, style, cutaway=False):
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
        if cutaway and in_cut(cx, cy) and not f["properties"].get("highlight"):
            continue  # don't leave buildings floating over the pit
        base = 0.0 if style == "schematic" else hm.sample(cx, cy) - 3.0
        target = bm_wh if f["properties"].get("highlight") else bm
        if not extrude_footprint(target, ring, base,
                                 f["properties"]["height"]):
            skipped += 1
        if f["properties"].get("highlight"):
            wilson = (cx, cy)

    for name, b in (("Buildings", bm), ("WilsonHall", bm_wh)):
        mesh = bpy.data.meshes.new(name + "Mesh")
        b.to_mesh(mesh)
        b.free()
        mesh.validate()
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.scene.collection.objects.link(obj)
        link_to(obj, "Buildings")
        if name == "WilsonHall":
            obj.data.materials.append(
                emission_material("WilsonHallMat", (0.00, 0.45, 0.70),
                                  strength=1.0, mix_principled=0.75))
        else:
            if style == "schematic":
                obj.data.materials.append(
                    principled_material("BuildingMat", (0.55, 0.57, 0.62)))
            else:
                obj.data.materials.append(
                    principled_material("BuildingMat", (0.42, 0.40, 0.37),
                                        roughness=0.9))
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
                          mix_principled=0.15))
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
                          mix_principled=0.15))
    link_to(obj, "Accelerators")
    return obj


def azimuth_point(ring_cfg, azimuth_deg):
    r = facility.ring_radius(ring_cfg)
    a = math.radians(azimuth_deg)
    cx, cy = ring_cfg["center"]
    return cx + r * math.cos(a), cy + r * math.sin(a)


def build_accelerators(hm, style):
    rings = {r["name"]: r for r in facility.RINGS}
    strength = 3.5 if style == "schematic" else 1.2
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
        cams.append(add_camera("Cam_WilsonHall",
                               (wx - 320, wy - 420, wz + 210),
                               (wx, wy, wz + 40), lens=50.0))
    bpy.context.scene.camera = cams[0]
    bpy.context.scene["render_cameras"] = [c.name for c in cams]


def build_lights(style):
    sun_data = bpy.data.lights.new("Sun", type="SUN")
    sun_data.energy = 3.0
    sun_data.angle = math.radians(1.0)
    sun = bpy.data.objects.new("Sun", sun_data)
    # from the southwest, ~40 deg elevation
    sun.rotation_euler = (math.radians(50.0), 0.0, math.radians(-135.0 + 180.0))
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
        sky.sun_elevation = math.radians(40.0)
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
    args = ap.parse_args(argv)

    # empty scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    hm = Heightmap(os.path.join(DATA, "heightmap.npz"))
    cutaway = (args.style == "realistic") and not args.no_cutaway

    build_terrain(hm, args.style, cutaway)
    build_boundary(hm, args.style)
    wilson_xy = build_buildings(hm, args.style, cutaway)
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
