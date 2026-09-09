"""The Fermilab site: prairie terrain, the Tevatron and Main Injector berms
with their cooling ponds, lakes, roads and street lights, ancillary
buildings, tree cover, distant town glow -- and the proposed 10 km muon
collider ring drawn as a luminous beamline with its two detector halls.

Coordinates are metres, x east, y north, z up, origin at Wilson Hall.
Positions are approximate but laid out to match the real site plan.
"""
from __future__ import annotations

import math
import random

import bpy

from . import common as C

# --- site plan (metres, relative to Wilson Hall) ---------------------------
TEV_C, TEV_R = (1150.0, 0.0), 1000.0                 # Tevatron main ring (6.3 km)
MI_C, MI_RX, MI_RY = (650.0, -1600.0), 560.0, 490.0    # Main Injector (3.3 km)
MC_C, MC_R = (1300.0, -1150.0), 1590.0                # proposed 10 km collider ring
MC_IP_ANGLES = (120.0, 300.0)                         # detector halls (deg, opposite); 120 deg sits NE of Wilson Hall, in frame
LINAC = ((-1000.0, -1600.0), (-225.0, -1600.0))       # proton driver -> ring (west crossing)

SITE_X = (-1000.0, 3450.0)   # Kirk Rd .. Eola Rd
SITE_Y = (-4000.0, 1800.0)   # south boundary .. Butterfield Rd

COLLIDER_COLOR = (0.25, 0.72, 1.0)
SODIUM = C.kelvin_rgb(2150)
LED_WHITE = C.kelvin_rgb(4200)


def _mc_point(angle_deg, z=0.0):
    a = math.radians(angle_deg)
    return (MC_C[0] + MC_R * math.cos(a), MC_C[1] + MC_R * math.sin(a), z)


def dist_to_ring(p, c, r):
    return abs(math.hypot(p[0] - c[0], p[1] - c[1]) - r)


# --------------------------------------------------------------------------- #
# terrain
# --------------------------------------------------------------------------- #
def prairie_material():
    """Grass004 x Ground037 blended by large-scale noise, tinted to late-summer
    tallgrass prairie, darkened for dusk."""
    mat, nt, bsdf, _ = C.new_material("prairie")
    g, d = C.tex_set("Grass004_2K-JPG"), C.tex_set("Ground037_2K-JPG")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    if not g.get("Color"):
        bsdf.inputs["Base Color"].default_value = (0.16, 0.14, 0.08, 1.0)
        bsdf.inputs["Roughness"].default_value = 1.0
        return mat
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1 / 7.0, 1 / 7.0, 1 / 7.0)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])

    def img(path, nc):
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = C.load_image(path, nc)
        n.projection = "FLAT"
        n.extension = "REPEAT"
        nt.links.new(mp.outputs["Vector"], n.inputs["Vector"])
        return n

    gc, dc = img(g["Color"], False), img(d["Color"], False)
    big = nt.nodes.new("ShaderNodeTexNoise")
    big.inputs["Scale"].default_value = 1 / 260.0
    big.inputs["Detail"].default_value = 3.0
    big.inputs["Roughness"].default_value = 0.6
    nt.links.new(tc.outputs["Object"], big.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.42
    ramp.color_ramp.elements[1].position = 0.62
    nt.links.new(big.outputs["Fac"], ramp.inputs["Fac"])
    _, mixed = C.mix_color(nt, ramp.outputs["Color"], gc.outputs["Color"], dc.outputs["Color"])
    # tallgrass tint (Sept: tan/gold with green) and dusk darkening
    _, tinted = C.mix_color(nt, 1.0, mixed, (0.80, 0.66, 0.42, 1.0), "MULTIPLY")
    nt.links.new(tinted, bsdf.inputs["Base Color"])
    bsdf.inputs["Specular IOR Level"].default_value = 0.12   # grass has almost no grazing sheen; keeps the far horizon dark
    if g.get("Roughness"):
        gr = img(g["Roughness"], True)
        nt.links.new(gr.outputs["Color"], bsdf.inputs["Roughness"])
    else:
        bsdf.inputs["Roughness"].default_value = 1.0
    if g.get("NormalGL"):
        gn = img(g["NormalGL"], True)
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.inputs["Strength"].default_value = 0.7
        nt.links.new(gn.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def build_terrain(col, mat):
    half = 30000.0
    C.mesh_object(
        "terrain",
        [(-half, -half, 0), (half, -half, 0), (half, half, 0), (-half, half, 0)],
        [(0, 1, 2, 3)],
        col=col,
        material=mat,
    )


# --------------------------------------------------------------------------- #
# accelerator rings
# --------------------------------------------------------------------------- #
BERM_PROFILE = [(-19.0, 0.0), (-7.0, 6.0), (7.0, 6.0), (19.0, 0.0)]


def build_rings(col, prairie, water):
    # Tevatron berm + inner cooling-pond ring
    C.ring_mesh("tevatron_berm", TEV_C, TEV_R, BERM_PROFILE, segments=480, col=col, material=prairie)
    C.ring_mesh("tevatron_cooling_ponds", TEV_C, TEV_R - 36.0, [(-12.0, 0.08), (12.0, 0.08)], segments=480, col=col, material=water, smooth=False)
    # Main Injector berm (slightly oval)
    C.ring_mesh("main_injector_berm", MI_C, MI_RX, BERM_PROFILE, ry=MI_RY, segments=360, col=col, material=prairie)
    # faint amber marker along the Tevatron crest: the reused tunnel
    tev_line = C.emissive_material("tevatron_tunnel_glow", (1.0, 0.62, 0.30), 0.5, camera_strength=1.6)
    C.tube_mesh("tevatron_tunnel_line", C.circle_points(TEV_C, TEV_R, n=480, z=6.6), 0.9, sides=6, col=col, material=tev_line, closed=True)


def build_collider(col, concrete):
    """The proposed muon collider: 10 km luminous ring, two detector halls,
    and the proton-driver linac feeding the western crossing."""
    core = C.emissive_material("collider_beam", COLLIDER_COLOR, 5.0, camera_strength=22.0)
    C.tube_mesh("muon_collider_ring", C.circle_points(MC_C, MC_R, n=900, z=2.2), 2.6, sides=12, col=col, material=core, closed=True)
    # soft outer sheath (dimmer, wider) gives the line body without blowing out
    sheath = C.emissive_material("collider_sheath", (0.45, 0.80, 1.0), 0.35, camera_strength=0.9)
    C.tube_mesh("muon_collider_sheath", C.circle_points(MC_C, MC_R, n=900, z=2.2), 5.0, sides=12, col=col, material=sheath, closed=True)

    hall_glow = C.emissive_material("hall_glow", COLLIDER_COLOR, 3.0, camera_strength=10.0)
    white = C.emissive_material("hall_white", LED_WHITE, 12.0)
    lamp_me = C.sphere_mesh_data("hall_lamp_m", 0.5)
    lamp_me.materials.append(white)
    for i, ang in enumerate(MC_IP_ANGLES):
        x, y, _ = _mc_point(ang)
        hall = C.bmesh_object(f"detector_hall_{i}", C.cylinder_bmesh(24.0, 26.0, n=64), col=col, material=concrete, location=(x, y, 0))
        C.ring_mesh(f"detector_hall_ring_{i}", (x, y), 27.5, [(-1.6, 0.3), (1.6, 0.3)], segments=96, col=col, material=hall_glow, smooth=False)
        C.ring_mesh(f"detector_hall_crown_{i}", (x, y), 22.5, [(-0.8, 26.2), (0.8, 26.2)], segments=96, col=col, material=hall_glow, smooth=False)
        C.box_object(f"detector_hall_annex_{i}", (60.0, 30.0, 12.0), (x + 55.0, y, 0), col=col, material=concrete)
        for k in range(6):
            a = 2 * math.pi * k / 6
            C.instance(f"hall_lamp_{i}_{k}", lamp_me, col=col, location=(x + 34 * math.cos(a), y + 34 * math.sin(a), 8.0))
        C.point_light(f"hall_light_{i}", (x, y, 40.0), power=30000, kelvin=6500, radius=6.0, col=col)

    linac = C.emissive_material("linac_beam", (0.85, 0.95, 1.0), 1.5, camera_strength=5.0)
    (x0, y0), (x1, y1) = LINAC
    C.tube_mesh("proton_driver_linac", [(x0, y0, 1.8), (x1, y1, 1.8)], 1.4, sides=8, col=col, material=linac)
    C.box_object("target_hall", (40.0, 18.0, 10.0), (x0 + 0.62 * (x1 - x0), y0 + 22.0, 0), col=col, material=concrete)


# --------------------------------------------------------------------------- #
# water
# --------------------------------------------------------------------------- #
LAKES = [
    # name, (x, y), rx, ry, rot(deg), seed
    ("lake_law", (-470.0, -800.0), 130.0, 90.0, 20, 1),
    ("swan_lake", (2350.0, -900.0), 160.0, 110.0, -15, 2),
    ("ae_sea", (2560.0, -380.0), 200.0, 150.0, 30, 3),
    ("caseys_pond", (330.0, 930.0), 115.0, 80.0, 10, 4),
    ("nepese_marsh", (-820.0, 320.0), 150.0, 95.0, -25, 5),
    ("lake_logo", (-60.0, 380.0), 70.0, 48.0, 0, 6),
    ("mi_pond_e", (1390.0, -1780.0), 105.0, 60.0, 15, 7),
    ("mi_pond_w", (-40.0, -1900.0), 90.0, 62.0, -10, 8),
    ("south_pond", (900.0, -2700.0), 140.0, 100.0, 40, 9),
    ("north_pond", (1600.0, 1350.0), 120.0, 75.0, -35, 10),
]


def build_water(col, water):
    for name, c, rx, ry, rot, seed in LAKES:
        C.blob_mesh(name, c, rx, ry=ry, z=0.08, rot=math.radians(rot), seed=seed, col=col, material=water)


# --------------------------------------------------------------------------- #
# roads and lights
# --------------------------------------------------------------------------- #
ROADS = {
    # name: polyline (or 'circle' spec)
    "pine_street": [(-1000.0, 62.0), (-330.0, 62.0), (-250.0, 55.0), (-175.0, 30.0), (-150.0, -10.0), (-150.0, -60.0)],
    "batavia_road": [(-1000.0, -1040.0), (3450.0, -1040.0)],
    "wilson_street": [(-1000.0, 1400.0), (3450.0, 1400.0)],
    "road_a": [(-260.0, -3900.0), (-260.0, 1750.0)],
    "road_d": [(2300.0, -3900.0), (2300.0, 1750.0)],
    "kautz_road": [(3000.0, -3900.0), (3000.0, 1750.0)],
    "kirk_road": [(-1000.0, -4000.0), (-1000.0, 1800.0)],
    "eola_road": [(3450.0, -4000.0), (3450.0, 1800.0)],
    "wh_south_drive": [(-150.0, -60.0), (-150.0, -260.0), (100.0, -260.0), (100.0, -500.0)],
    "wh_east_drive": [(240.0, 120.0), (240.0, -120.0)],
}


def asphalt_material():
    return C.pbr_material("asphalt", "Asphalt031_1K-JPG", scale=6.0, tint=(0.55, 0.55, 0.58), roughness_mult=0.55, normal_strength=0.5, projection="FLAT")


def build_roads(col, asphalt):
    for name, pts in ROADS.items():
        C.ribbon_mesh(name, pts, 8.0, 0.12, col=col, material=asphalt)
    C.ribbon_mesh("ring_road", C.circle_points(TEV_C, TEV_R + 26.0, n=480), 6.5, 0.12, col=col, material=asphalt, closed=True)
    C.ribbon_mesh("mi_road", C.circle_points(MI_C, MI_RX + 26.0, n=360, ry=MI_RY + 26.0), 6.5, 0.12, col=col, material=asphalt, closed=True)


def _lamp_factory(col):
    sodium = C.emissive_material("lamp_sodium", SODIUM, 22.0, camera_strength=24.0)
    led = C.emissive_material("lamp_led", LED_WHITE, 16.0, camera_strength=20.0)
    pole = C.flat_material("lamp_pole", (0.05, 0.05, 0.05), roughness=0.6)
    head = C.sphere_mesh_data("lamp_head", 0.42)
    pole_bm = C.cylinder_bmesh(0.12, 9.0, n=6)
    pole_me = bpy.data.meshes.new("lamp_pole_m")
    pole_bm.to_mesh(pole_me)
    pole_bm.free()
    pole_me.materials.append(pole)
    head_s = head.copy()
    head_s.materials.append(sodium)
    head_l = head.copy()
    head_l.materials.append(led)
    bpy.data.meshes.remove(head)
    counter = [0]

    jit = random.Random(23)

    def lamp(x, y, kind="sodium"):
        counter[0] += 1
        x += jit.uniform(-1.5, 1.5)
        y += jit.uniform(-1.5, 1.5)
        C.instance(f"lamp_pole_{counter[0]}", pole_me, col=col, location=(x, y, 0.1))
        C.instance(f"lamp_{counter[0]}", head_s if kind == "sodium" else head_l, col=col, location=(x, y, 9.2))

    return lamp


def build_street_lights(col):
    lamp = _lamp_factory(col)
    # Pine Street approach, alternating sides
    for i, x in enumerate(range(-1000, -330, 46)):
        lamp(x, 62.0 + (7.0 if i % 2 else -7.0), "sodium")
    # Wilson Hall west parking lot grid
    for x in range(-330, -120, 42):
        for y in range(-105, 106, 52):
            lamp(x + (21 if (y // 52) % 2 else 0), y, "led" if (x + y) % 3 == 0 else "sodium")
    # plaza / pond edge
    for y in (-40.0, 0.0, 40.0):
        lamp(70.0, y, "led")
        lamp(190.0, y, "led")
    # south drive + IARC lot
    for y in range(-90, -480, -45):
        lamp(-157.0, y, "sodium")
    for x in range(120, 300, 40):
        lamp(x, -180.0, "led")
    # Batavia Road near the Road A / Kautz intersections and Road A near WH
    for x in range(-500, 520, 62):
        lamp(x, -1047.0, "sodium")
    for y in range(-700, 700, 60):
        lamp(-267.0, y, "sodium")
    return lamp


# --------------------------------------------------------------------------- #
# buildings
# --------------------------------------------------------------------------- #
def _lit_box_material(name, base, *, lit_color, lit_strength=5.0, band=(0.35, 0.75), pane_w=2.4, lit_fraction=0.55, seed=1.0):
    """Windows on a generic building: lit panes by hash, else dark facade."""
    mat, nt, bsdf, out = C.new_material(name)
    bsdf.inputs["Base Color"].default_value = (*base, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.75
    tc = nt.nodes.new("ShaderNodeTexCoord")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    pos = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Generated"], pos.inputs["Vector"])  # 0..1 over the box
    nrm = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Normal"], nrm.inputs["Vector"])
    zf = C.nmath(nt, "MULTIPLY", pos.outputs["Z"], value_b=1.0)  # single band per storey handled by scale
    # storeys: assume 4 m storeys, encoded via object scale is unknown here, so use 3 bands
    zz = C.nmath(nt, "MULTIPLY", pos.outputs["Z"], value_b=3.0)
    fr = C.nmath(nt, "FRACT", zz.outputs[0])
    lo = C.nmath(nt, "GREATER_THAN", fr.outputs[0], value_b=band[0])
    hi = C.nmath(nt, "LESS_THAN", fr.outputs[0], value_b=band[1])
    bandm = C.nmath(nt, "MULTIPLY", lo.outputs[0], hi.outputs[0])
    nz = C.nmath(nt, "ABSOLUTE", nrm.outputs["Z"])
    wall = C.nmath(nt, "LESS_THAN", nz.outputs[0], value_b=0.5)
    # pane hash
    comb = nt.nodes.new("ShaderNodeCombineXYZ")
    fl = C.nmath(nt, "FLOOR", zz.outputs[0])
    ux = C.nmath(nt, "MULTIPLY", pos.outputs["X"], value_b=14.0)
    uy = C.nmath(nt, "MULTIPLY", pos.outputs["Y"], value_b=14.0)
    _, u = C.mix_float(nt, C.nmath(nt, "ABSOLUTE", nrm.outputs["Y"]).outputs[0], uy.outputs[0], ux.outputs[0])
    ufl = C.nmath(nt, "FLOOR", u)
    nt.links.new(fl.outputs[0], comb.inputs["X"])
    nt.links.new(ufl.outputs[0], comb.inputs["Y"])
    sgn = C.nmath(nt, "MULTIPLY_ADD", nrm.outputs["X"], value_b=3.0)
    sgn.inputs[2].default_value = seed
    sgn2 = C.nmath(nt, "MULTIPLY_ADD", nrm.outputs["Y"], value_b=5.0)
    nt.links.new(sgn.outputs[0], sgn2.inputs[2])
    nt.links.new(sgn2.outputs[0], comb.inputs["Z"])
    wn = nt.nodes.new("ShaderNodeTexWhiteNoise")
    wn.noise_dimensions = "3D"
    nt.links.new(comb.outputs["Vector"], wn.inputs["Vector"])
    lit = C.nmath(nt, "GREATER_THAN", wn.outputs["Value"], value_b=1.0 - lit_fraction)
    ufr = C.nmath(nt, "FRACT", u)
    gap = C.nmath(nt, "GREATER_THAN", ufr.outputs[0], value_b=0.12)
    win = C.nmath(nt, "MULTIPLY", bandm.outputs[0], wall.outputs[0])
    win = C.nmath(nt, "MULTIPLY", win.outputs[0], lit.outputs[0])
    win = C.nmath(nt, "MULTIPLY", win.outputs[0], gap.outputs[0])
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*lit_color, 1.0)
    em.inputs["Strength"].default_value = lit_strength
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(win.outputs[0], mix.inputs["Fac"])
    nt.links.new(bsdf.outputs["BSDF"], mix.inputs[1])
    nt.links.new(em.outputs["Emission"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return mat


BUILDINGS = [
    # name, (x, y), (sx, sy, sz), rot_deg, kind
    ("iarc", (150.0, -150.0), (70.0, 42.0, 15.0), 0, "glass"),
    ("feynman_computing", (260.0, 540.0), (52.0, 34.0, 13.0), 0, "office"),
    ("icb_1", (-140.0, -520.0), (120.0, 42.0, 10.0), 0, "industrial"),
    ("icb_2", (-140.0, -600.0), (120.0, 42.0, 10.0), 0, "industrial"),
    ("icb_3", (-140.0, -680.0), (110.0, 40.0, 10.0), 0, "industrial"),
    ("technical_division", (-330.0, -560.0), (90.0, 45.0, 11.0), 0, "industrial"),
    ("mc1_gminus2", (230.0, -800.0), (62.0, 32.0, 12.0), 0, "office"),
    ("mu2e_hall", (360.0, -900.0), (52.0, 30.0, 11.0), 0, "industrial"),
    ("meson_lab", (-320.0, 1180.0), (200.0, 32.0, 10.0), 0, "industrial"),
    ("neutrino_line", (1450.0, 720.0), (150.0, 26.0, 8.0), -18, "industrial"),
    ("cdf_hall", (650.0, 866.0), (48.0, 62.0, 22.0), 0, "industrial"),
    ("dzero_hall", (1150.0 - 500.0, -866.0), (48.0, 62.0, 22.0), 0, "industrial"),
    ("cryo_plant", (1150.0 + 1030.0, 40.0), (60.0, 28.0, 9.0), 0, "industrial"),
    ("mi_8_service", (650.0 + 400.0, -1600.0 - 520.0), (36.0, 18.0, 6.0), 0, "industrial"),
    ("mi_60_service", (650.0 - 560.0 - 30.0, -1600.0), (36.0, 18.0, 6.0), 0, "industrial"),
    ("site_38", (-620.0, -300.0), (44.0, 30.0, 8.0), 0, "office"),
    ("lederman_center", (-720.0, -1050.0 + 60.0), (30.0, 20.0, 7.0), 0, "office"),
]


def build_buildings(col, concrete):
    kinds = {
        "glass": _lit_box_material("bld_glass", (0.06, 0.07, 0.08), lit_color=C.kelvin_rgb(4300), lit_strength=0.9, band=(0.2, 0.85), lit_fraction=0.45, seed=3.0),
        "office": _lit_box_material("bld_office", (0.30, 0.28, 0.25), lit_color=C.kelvin_rgb(3300), lit_strength=1.8, lit_fraction=0.55, seed=5.0),
        "industrial": _lit_box_material("bld_industrial", (0.22, 0.22, 0.21), lit_color=C.kelvin_rgb(3800), lit_strength=1.2, band=(0.55, 0.8), lit_fraction=0.25, seed=9.0),
    }
    unit = C.box_mesh_data("unit_box")
    for name, (x, y), size, rot, kind in BUILDINGS:
        me = unit.copy()
        me.materials.append(kinds[kind])
        C.instance(name, me, col=col, location=(x, y, 0.0), rotation=(0, 0, math.radians(rot)), scale=size)
    # Tevatron sector service buildings (A0..F0) with a white lamp each
    lamp_mat = C.emissive_material("service_lamp", LED_WHITE, 15.0, camera_strength=30.0)
    lamp_me = C.sphere_mesh_data("service_lamp_m", 0.45)
    lamp_me.materials.append(lamp_mat)
    for k in range(6):
        a = math.radians(60 * k + 15)
        r = TEV_R + 62.0
        x, y = TEV_C[0] + r * math.cos(a), TEV_C[1] + r * math.sin(a)
        me = unit.copy()
        me.materials.append(kinds["industrial"])
        C.instance(f"tev_service_{k}", me, col=col, location=(x, y, 0.0), rotation=(0, 0, a), scale=(28.0, 14.0, 6.0))
        C.instance(f"tev_service_lamp_{k}", lamp_me, col=col, location=(x + 12 * math.cos(a + 1.2), y + 12 * math.sin(a + 1.2), 8.0))
    # Fermilab Village (former farmhouses, east side): warm scattered dwellings
    rng = random.Random(7)
    house = _lit_box_material("bld_house", (0.35, 0.30, 0.26), lit_color=C.kelvin_rgb(2900), lit_strength=1.5, band=(0.3, 0.7), lit_fraction=0.6, seed=11.0)
    for i in range(34):
        x = 2850.0 + rng.uniform(-260, 260)
        y = 250.0 + rng.uniform(-220, 220)
        me = unit.copy()
        me.materials.append(house)
        C.instance(f"village_{i}", me, col=col, location=(x, y, 0.0), rotation=(0, 0, rng.uniform(0, math.pi)), scale=(rng.uniform(10, 18), rng.uniform(9, 14), rng.uniform(5, 8)))


# --------------------------------------------------------------------------- #
# trees
# --------------------------------------------------------------------------- #
def _keep_out(x, y):
    """Avoid roads, berms, rings, water and Wilson Hall when scattering trees."""
    if -140 < x < 330 and -110 < y < 110:            # Wilson Hall + plaza + pond
        return False
    if dist_to_ring((x, y), TEV_C, TEV_R) < 60:
        return False
    if abs(math.hypot((x - MI_C[0]) / MI_RX, (y - MI_C[1]) / MI_RY) - 1.0) * 500 < 55:
        return False
    if dist_to_ring((x, y), MC_C, MC_R) < 25:
        return False
    for pts in ROADS.values():
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            dx, dy = x1 - x0, y1 - y0
            t = max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / (dx * dx + dy * dy + 1e-9)))
            if math.hypot(x - (x0 + t * dx), y - (y0 + t * dy)) < 10:
                return False
    for _, c, rx, ry, rot, _ in LAKES:
        if math.hypot((x - c[0]) / (rx * 1.3), (y - c[1]) / (ry * 1.3)) < 1.0:
            return False
    for _, (bx, by), (sx, sy, _), _, _ in BUILDINGS:
        if abs(x - bx) < sx / 2 + 8 and abs(y - by) < sy / 2 + 8:
            return False
    return True


WOODS = [
    # centre, rx, ry, count  (Big Woods west/north-west, riparian strips, east woods)
    ((-620.0, 900.0), 420.0, 330.0, 1100),
    ((-780.0, -1900.0), 260.0, 420.0, 520),
    ((2650.0, 520.0), 380.0, 260.0, 520),
    ((2800.0, -1500.0), 300.0, 260.0, 380),
    ((1000.0, 1450.0), 300.0, 160.0, 260),
    ((-400.0, 380.0), 170.0, 110.0, 160),
    ((-150.0, -1350.0), 220.0, 120.0, 180),
    ((400.0, -2600.0), 500.0, 250.0, 420),
    ((-420.0, -230.0), 110.0, 70.0, 140),     # copse SW of Wilson Hall: dark foreground anchor for the aerial camera
]


def build_trees(col, *, density=1.0):
    rng = random.Random(11)
    canopy_mat = C.flat_material("canopy", (0.035, 0.055, 0.025), roughness=0.9)
    trunk_mat = C.flat_material("trunk", (0.05, 0.04, 0.03), roughness=0.9)
    canopies = [C.canopy_mesh_data(f"canopy_{i}", 100 + i) for i in range(4)]
    for m in canopies:
        m.materials.append(canopy_mat)
    trunk_bm = C.cylinder_bmesh(0.35, 5.0, n=6)
    trunk = bpy.data.meshes.new("trunk_m")
    trunk_bm.to_mesh(trunk)
    trunk_bm.free()
    trunk.materials.append(trunk_mat)
    n = [0]

    def tree(x, y, s=None):
        s = s or rng.uniform(5.5, 9.5)
        n[0] += 1
        C.instance(f"tree_{n[0]}", canopies[n[0] % 4], col=col, location=(x, y, 4.2 + 0.55 * s), rotation=(0, 0, rng.uniform(0, 6.28)), scale=(s, s * rng.uniform(0.85, 1.1), s * 0.9))
        C.instance(f"trunk_{n[0]}", trunk, col=col, location=(x, y, 0.0), scale=(1, 1, 0.9 + 0.1 * s))

    for c, rx, ry, count in WOODS:
        for _ in range(int(count * density)):
            a, r = rng.uniform(0, 2 * math.pi), math.sqrt(rng.random())
            x, y = c[0] + rx * r * math.cos(a), c[1] + ry * r * math.sin(a)
            if _keep_out(x, y):
                tree(x, y, rng.uniform(6.0, 10.5))
    # tree lines along Pine Street and Road A
    for x in range(-1000, -330, 13):
        for side in (-1, 1):
            if rng.random() < 0.85:
                tree(x + rng.uniform(-2, 2), 62.0 + side * 16.0 + rng.uniform(-2, 2), rng.uniform(5, 8))
    for y in range(-1900, 1600, 15):
        if rng.random() < 0.6:
            tree(-260.0 + rng.choice((-1, 1)) * 17.0 + rng.uniform(-2, 2), y + rng.uniform(-3, 3), rng.uniform(5, 8))
    # scattered prairie trees / hedgerows
    for _ in range(int(650 * density)):
        x, y = rng.uniform(*SITE_X), rng.uniform(*SITE_Y)
        if math.hypot(x - TEV_C[0], y - TEV_C[1]) < 930:    # restored prairie inside the ring
            continue
        if _keep_out(x, y):
            tree(x, y)
    # clumps around Wilson Hall's north side and the lakes
    for _ in range(int(90 * density)):
        x, y = rng.uniform(-230, 60), rng.uniform(120, 300)
        if _keep_out(x, y):
            tree(x, y, rng.uniform(5, 8))
    for _, c, rx, ry, rot, _ in LAKES:
        for _ in range(int(28 * density)):
            a = rng.uniform(0, 2 * math.pi)
            x, y = c[0] + rx * 1.45 * math.cos(a), c[1] + ry * 1.45 * math.sin(a)
            if _keep_out(x, y):
                tree(x, y, rng.uniform(5, 8))
    return n[0]


# --------------------------------------------------------------------------- #
# distant towns: horizon glow
# --------------------------------------------------------------------------- #
def _city_material(name, color, strength, scale):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.remove(nt.nodes["Principled BSDF"])
    out = nt.nodes["Material Output"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    n1 = nt.nodes.new("ShaderNodeTexNoise")
    n1.inputs["Scale"].default_value = scale
    n1.inputs["Detail"].default_value = 6.0
    n1.inputs["Roughness"].default_value = 0.7
    nt.links.new(tc.outputs["Object"], n1.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.5
    ramp.color_ramp.elements[1].position = 0.75
    nt.links.new(n1.outputs["Fac"], ramp.inputs["Fac"])
    # fade to nothing towards the top of the strip
    pos = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Generated"], pos.inputs["Vector"])
    fade = C.nmath(nt, "SUBTRACT", value_a=1.0, b=pos.outputs["Z"])
    fade = C.nmath(nt, "POWER", fade.outputs[0], value_b=2.2)
    s = C.nmath(nt, "MULTIPLY", ramp.outputs["Color"], fade.outputs[0])
    s = C.nmath(nt, "MULTIPLY", s.outputs[0], value_b=strength)
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*color, 1.0)
    nt.links.new(s.outputs[0], em.inputs["Strength"])
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat


def build_towns(col):
    amber = (1.0, 0.58, 0.28)
    # Naperville / Warrenville (east), Batavia-Geneva (north-west), Aurora (south)
    strips = [
        ("towns_east", (9500.0, -1500.0), 26000.0, 70.0, math.radians(90), 0.45, 0.0011),
        ("towns_north", (-2500.0, 5200.0), 14000.0, 45.0, 0.0, 0.32, 0.0016),
        ("towns_south", (1500.0, -8500.0), 16000.0, 45.0, 0.0, 0.28, 0.0014),
    ]
    for name, (x, y), length, height, rot, strength, scale in strips:
        mat = _city_material(name + "_m", amber, strength, scale)
        obj = C.box_object(name, (length, 30.0, height), (x, y, 0.0), rot_z=rot, col=col, material=mat)
        obj.visible_shadow = False
