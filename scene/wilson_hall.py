"""Robert R. Wilson Hall -- Fermilab's 16-storey twin-tower high-rise.

Two concrete towers, separated north-south, lean inward along catenary-like
curves until they touch at the roof; the slot between them is the glazed
atrium (open east and west). The long faces carry continuous ribbon windows.
All geometry is generated from the profile functions below; the facade is a
procedural material (floor bands, panes, randomly lit offices) layered over a
board-formed concrete texture set.
"""
from __future__ import annotations

import math

import bmesh
import bpy

from . import common as C

FLOORS = 16
FLOOR_H = 4.5
HEIGHT = FLOORS * FLOOR_H            # 72 m to the roof slab
LENGTH = 78.0                        # east-west extent of each tower
STEPS = 56                           # vertical resolution of the lofted profile


def y_outer(t: float) -> float:
    """Outer (north) face of the north tower vs normalised height t in [0, 1]."""
    return 34.0 - 25.0 * t ** 2.3


def thickness(t: float) -> float:
    return 18.0 - 9.0 * t


def y_inner(t: float) -> float:
    return y_outer(t) - thickness(t)


# --------------------------------------------------------------------------- #
# materials
# --------------------------------------------------------------------------- #
def facade_material(concrete_tex="Concrete034_2K-JPG", *, lit_fraction=0.80, window_emission=2.2):
    """Concrete + ribbon windows driven by object-space position.

    Window band: 22 %..75 % of each floor height. Panes 3 m wide with thin
    mullions. Each pane is lit (warm, randomly 2700-4200 K) with probability
    `lit_fraction`; unlit panes are dark glossy glass reflecting the sky.
    """
    concrete = C.pbr_material("wh_concrete_src", concrete_tex, scale=3.5, tint=(0.92, 0.88, 0.80), roughness_mult=1.0, normal_strength=0.8)
    mat, nt, bsdf, out = C.new_material("wilson_facade")

    # --- bring the concrete PBR graph into this material -------------------
    # (simplest: rebuild the same graph here via the helper on this tree)
    nt.nodes.remove(bsdf)
    conc_bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    maps = C.tex_set(concrete_tex)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    if maps.get("Color"):
        mp = nt.nodes.new("ShaderNodeMapping")
        mp.inputs["Scale"].default_value = (1 / 3.5, 1 / 3.5, 1 / 3.5)
        nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])

        def img(path, nc):
            n = nt.nodes.new("ShaderNodeTexImage")
            n.image = C.load_image(path, nc)
            n.projection = "BOX"
            n.projection_blend = 0.3
            nt.links.new(mp.outputs["Vector"], n.inputs["Vector"])
            return n

        col = img(maps["Color"], False)
        _, tinted = C.mix_color(nt, 1.0, col.outputs["Color"], (0.92, 0.88, 0.80, 1.0), "MULTIPLY")
        nt.links.new(tinted, conc_bsdf.inputs["Base Color"])
        if maps.get("Roughness"):
            r = img(maps["Roughness"], True)
            nt.links.new(r.outputs["Color"], conc_bsdf.inputs["Roughness"])
        if maps.get("NormalGL"):
            nrm = img(maps["NormalGL"], True)
            nm = nt.nodes.new("ShaderNodeNormalMap")
            nm.inputs["Strength"].default_value = 0.8
            nt.links.new(nrm.outputs["Color"], nm.inputs["Color"])
            nt.links.new(nm.outputs["Normal"], conc_bsdf.inputs["Normal"])
    else:
        conc_bsdf.inputs["Base Color"].default_value = (0.55, 0.52, 0.47, 1.0)
        conc_bsdf.inputs["Roughness"].default_value = 0.9
    bpy.data.materials.remove(concrete)

    # --- window mask from object-space position ------------------------------
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    pos = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], pos.inputs["Vector"])
    nrm = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Normal"], nrm.inputs["Vector"])

    zf = C.nmath(nt, "DIVIDE", pos.outputs["Z"], value_b=FLOOR_H)
    floor_idx = C.nmath(nt, "FLOOR", zf.outputs[0])
    frac = C.nmath(nt, "FRACT", zf.outputs[0])
    band_lo = C.nmath(nt, "GREATER_THAN", frac.outputs[0], value_b=0.22)
    band_hi = C.nmath(nt, "LESS_THAN", frac.outputs[0], value_b=0.75)
    band = C.nmath(nt, "MULTIPLY", band_lo.outputs[0], band_hi.outputs[0])

    # facade-parallel coordinate: x on N/S faces, y on E/W ends
    ny_abs = C.nmath(nt, "ABSOLUTE", nrm.outputs["Y"])
    is_ns = C.nmath(nt, "GREATER_THAN", ny_abs.outputs[0], value_b=0.5)
    _, u = C.mix_float(nt, is_ns.outputs[0], pos.outputs["Y"], pos.outputs["X"])
    uf = C.nmath(nt, "DIVIDE", u, value_b=3.0)
    pane_idx = C.nmath(nt, "FLOOR", uf.outputs[0])
    ufr = C.nmath(nt, "FRACT", uf.outputs[0])
    m_lo = C.nmath(nt, "GREATER_THAN", ufr.outputs[0], value_b=0.035)
    m_hi = C.nmath(nt, "LESS_THAN", ufr.outputs[0], value_b=0.965)
    pane_open = C.nmath(nt, "MULTIPLY", m_lo.outputs[0], m_hi.outputs[0])

    # ends (E/W) get sparse windows; roof none
    nx_abs = C.nmath(nt, "ABSOLUTE", nrm.outputs["X"])
    is_end = C.nmath(nt, "GREATER_THAN", nx_abs.outputs[0], value_b=0.5)
    nz_up = C.nmath(nt, "GREATER_THAN", nrm.outputs["Z"], value_b=0.5)
    not_roof = C.nmath(nt, "SUBTRACT", value_a=1.0, b=nz_up.outputs[0])

    # per-pane hash
    comb = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(floor_idx.outputs[0], comb.inputs["X"])
    nt.links.new(pane_idx.outputs[0], comb.inputs["Y"])
    nt.links.new(is_ns.outputs[0], comb.inputs["Z"])
    side = C.nmath(nt, "MULTIPLY_ADD", nrm.outputs["Y"], value_b=7.0)
    side.inputs[2].default_value = 13.0
    comb2 = nt.nodes.new("ShaderNodeVectorMath")
    comb2.operation = "ADD"
    nt.links.new(comb.outputs["Vector"], comb2.inputs[0])
    comb_side = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(side.outputs[0], comb_side.inputs["X"])
    nt.links.new(nrm.outputs["X"], comb_side.inputs["Y"])
    nt.links.new(comb_side.outputs["Vector"], comb2.inputs[1])
    wn = nt.nodes.new("ShaderNodeTexWhiteNoise")
    wn.noise_dimensions = "3D"
    nt.links.new(comb2.outputs["Vector"], wn.inputs["Vector"])
    wn2 = nt.nodes.new("ShaderNodeTexWhiteNoise")
    wn2.noise_dimensions = "4D"
    nt.links.new(comb2.outputs["Vector"], wn2.inputs["Vector"])
    wn2.inputs["W"].default_value = 7.31

    # end walls: only ~25 % of panes are windows at all
    end_keep = C.nmath(nt, "GREATER_THAN", wn2.outputs["Value"], value_b=0.75)
    _, pane_exists = C.mix_float(nt, is_end.outputs[0], 1.0, end_keep.outputs[0])

    window = C.nmath(nt, "MULTIPLY", band.outputs[0], pane_open.outputs[0])
    window = C.nmath(nt, "MULTIPLY", window.outputs[0], pane_exists)
    window = C.nmath(nt, "MULTIPLY", window.outputs[0], not_roof.outputs[0])

    # whole floors are lit or dark (hash of floor+side), then a few panes per lit floor are off
    comb_f = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(floor_idx.outputs[0], comb_f.inputs["X"])
    nt.links.new(side.outputs[0], comb_f.inputs["Y"])
    nt.links.new(nrm.outputs["X"], comb_f.inputs["Z"])
    wn_f = nt.nodes.new("ShaderNodeTexWhiteNoise")
    wn_f.noise_dimensions = "3D"
    nt.links.new(comb_f.outputs["Vector"], wn_f.inputs["Vector"])
    floor_lit = C.nmath(nt, "GREATER_THAN", wn_f.outputs["Value"], value_b=0.22)
    pane_lit = C.nmath(nt, "GREATER_THAN", wn.outputs["Value"], value_b=1.0 - lit_fraction)
    lit = C.nmath(nt, "MULTIPLY", floor_lit.outputs[0], pane_lit.outputs[0])
    # brightness + colour temperature per office
    bright = C.nmath(nt, "MULTIPLY_ADD", wn2.outputs["Value"], value_b=0.9)
    bright.inputs[2].default_value = 0.45
    kel = C.nmath(nt, "MULTIPLY_ADD", wn.outputs["Value"], value_b=1500.0)
    kel.inputs[2].default_value = 2700.0
    warm = C.blackbody(nt, kel.outputs[0])

    # glass BSDF (unlit) and emission (lit)
    glass = nt.nodes.new("ShaderNodeBsdfPrincipled")
    glass.inputs["Base Color"].default_value = (0.02, 0.03, 0.035, 1.0)
    glass.inputs["Roughness"].default_value = 0.06
    glass.inputs["IOR"].default_value = 1.5
    glass.inputs["Specular IOR Level"].default_value = 0.6
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(warm, em.inputs["Color"])
    strength = C.nmath(nt, "MULTIPLY", bright.outputs[0], value_b=window_emission)
    nt.links.new(strength.outputs[0], em.inputs["Strength"])
    # lit windows: emission + a little glossy reflection on top
    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(em.outputs["Emission"], add.inputs[0])
    nt.links.new(glass.outputs["BSDF"], add.inputs[1])
    mix_lit = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lit.outputs[0], mix_lit.inputs["Fac"])
    nt.links.new(glass.outputs["BSDF"], mix_lit.inputs[1])
    nt.links.new(add.outputs["Shader"], mix_lit.inputs[2])
    mix_win = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(window.outputs[0], mix_win.inputs["Fac"])
    nt.links.new(conc_bsdf.outputs["BSDF"], mix_win.inputs[1])
    nt.links.new(mix_lit.outputs["Shader"], mix_win.inputs[2])
    nt.links.new(mix_win.outputs["Shader"], out.inputs["Surface"])
    return mat


def atrium_glass_material():
    mat, nt, bsdf, _ = C.new_material("wh_atrium_glass")
    bsdf.inputs["Base Color"].default_value = (0.80, 0.90, 0.88, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.03
    bsdf.inputs["IOR"].default_value = 1.5
    bsdf.inputs["Transmission Weight"].default_value = 1.0
    bsdf.inputs["Thin Wall"].default_value = True
    mat.use_transparent_shadow = True
    return mat


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def _tower(name, sign, mat, col):
    bm = bmesh.new()
    rows = []
    for k in range(STEPS + 1):
        t = k / STEPS
        z = t * HEIGHT
        yo, yi = sign * y_outer(t), sign * y_inner(t)
        rows.append([
            bm.verts.new((-LENGTH / 2, yi, z)),
            bm.verts.new((LENGTH / 2, yi, z)),
            bm.verts.new((LENGTH / 2, yo, z)),
            bm.verts.new((-LENGTH / 2, yo, z)),
        ])
    for k in range(STEPS):
        a, b = rows[k], rows[k + 1]
        for j in range(4):
            bm.faces.new((a[j], a[(j + 1) % 4], b[(j + 1) % 4], b[j]))
    bm.faces.new(rows[-1])  # roof
    return C.bmesh_object(name, bm, col=col, material=mat, smooth=False)


def _atrium_glass(name, x, mat, col):
    """Glazed end wall filling the gap between the towers at x = +-L/2."""
    bm = bmesh.new()
    rows = []
    for k in range(STEPS + 1):
        t = k / STEPS
        yi = max(y_inner(t), 0.05)
        rows.append((bm.verts.new((x, -yi, t * HEIGHT)), bm.verts.new((x, yi, t * HEIGHT))))
    for k in range(STEPS):
        bm.faces.new((rows[k][0], rows[k][1], rows[k + 1][1], rows[k + 1][0]))
    return C.bmesh_object(name, bm, col=col, material=mat, smooth=True)


def _balconies(col):
    """Lit floor edges inside the atrium (what you see glowing through the glass)."""
    mat = C.emissive_material("wh_balcony_glow", C.kelvin_rgb(3000), 3.5)
    bm = bmesh.new()
    for f in range(1, FLOORS):
        z = f * FLOOR_H + 0.25
        t = z / HEIGHT
        yi = y_inner(t) - 0.2
        if yi < 1.0:
            continue
        for x0, x1 in ((-LENGTH / 2 + 4.0, -LENGTH / 2 + 5.4), (LENGTH / 2 - 5.4, LENGTH / 2 - 4.0)):
            v = [bm.verts.new((x0, -yi, z)), bm.verts.new((x1, -yi, z)), bm.verts.new((x1, yi, z)), bm.verts.new((x0, yi, z))]
            bm.faces.new(v)
            v2 = [bm.verts.new((x0, -yi, z + 0.5)), bm.verts.new((x1, -yi, z + 0.5)), bm.verts.new((x1, yi, z + 0.5)), bm.verts.new((x0, yi, z + 0.5))]
            bm.faces.new(v2)
    return C.bmesh_object("wh_balconies", bm, col=col, material=mat)


def build(col: bpy.types.Collection, *, floodlights=True):
    """Build Wilson Hall (+ Ramsey Auditorium, plaza, reflecting pond, obelisk)
    centred on the origin. Returns the north tower (used as camera focus)."""
    facade = facade_material()
    glass = atrium_glass_material()
    concrete_dark = C.pbr_material("wh_concrete_dark", "Concrete042A_1K-JPG", scale=3.0, tint=(0.75, 0.72, 0.68), normal_strength=0.6)
    plaza_mat = C.pbr_material("wh_plaza", "Concrete042A_1K-JPG", scale=2.5, tint=(0.55, 0.53, 0.50), roughness_mult=0.9, normal_strength=0.4)

    north = _tower("wilson_hall_north", +1, facade, col)
    _tower("wilson_hall_south", -1, facade, col)
    _atrium_glass("wh_atrium_glass_e", LENGTH / 2 - 0.3, glass, col)
    _atrium_glass("wh_atrium_glass_w", -LENGTH / 2 + 0.3, glass, col)
    _balconies(col)

    # roof slab joining the towers + mechanical penthouse
    C.box_object("wh_roof_slab", (LENGTH + 0.6, 2 * y_outer(1.0) + 0.8, 3.2), (0, 0, HEIGHT), col=col, material=concrete_dark)
    C.box_object("wh_penthouse", (30.0, 10.0, 3.0), (0, 0, HEIGHT + 3.2), col=col, material=concrete_dark)

    # Ramsey Auditorium: fan-shaped hall attached to the west end
    bm = bmesh.new()
    cx = -LENGTH / 2 - 26.0
    r = 36.0
    base = [bm.verts.new((cx, -14.0, 0)), bm.verts.new((cx, 14.0, 0))]
    arc = [bm.verts.new((cx + r * math.cos(a), r * math.sin(a), 0)) for a in [math.radians(110 + 140 * i / 24) for i in range(25)]]
    bottom = [base[1]] + arc + [base[0]]
    top = [bm.verts.new((v.co.x, v.co.y, 11.0)) for v in bottom]
    bm.faces.new(bottom[::-1])
    bm.faces.new(top)
    for i in range(len(bottom) - 1):
        bm.faces.new((bottom[i], bottom[i + 1], top[i + 1], top[i]))
    bm.faces.new((bottom[-1], bottom[0], top[0], top[-1]))
    C.bmesh_object("ramsey_auditorium", bm, col=col, material=concrete_dark)
    C.box_object("wh_west_link", (28.0, 30.0, 9.0), (-LENGTH / 2 - 14.0, 0, 0), col=col, material=concrete_dark)

    # plaza + reflecting pond (east) + hyperbolic obelisk
    C.box_object("wh_plaza", (320.0, 170.0, 0.25), (40.0, 0.0, -0.05), col=col, material=plaza_mat)
    water = C.water_material("wh_pond_water", ripple_scale=2.0, ripple_strength=0.04)
    C.box_object("wh_pond_curb", (128.0, 66.0, 0.6), (120.0, 0.0, 0.0), col=col, material=concrete_dark)
    bm = C.box_bmesh(124.0, 62.0, 0.2)
    C.bmesh_object("wh_reflecting_pond", bm, col=col, material=water, location=(120.0, 0.0, 0.45))
    obelisk_mat = C.flat_material("obelisk", (0.9, 0.9, 0.9), roughness=0.4)
    ob = C.bmesh_object("hyperbolic_obelisk", C.cone_bmesh(1.4, 0.05, 11.0, n=3), col=col, material=obelisk_mat, location=(196.0, 0.0, 0.2))
    C.spot_light("obelisk_spot", (191.0, -6.0, 0.4), (196.0, 0.0, 6.0), power=6000, kelvin=4000, size_deg=35, blend=0.4, radius=0.2, col=col)

    if floodlights:
        # warm architectural floods raking up the south face and the west end
        for i, (x, y) in enumerate(((-24.0, -78.0), (24.0, -78.0))):
            C.spot_light(f"wh_flood_s{i}", (x, y, 1.0), (x, -22.0, 42.0), power=42000, kelvin=3000, size_deg=58, blend=0.55, radius=0.6, col=col)
        for i, y in enumerate((-20.0, 20.0)):
            C.spot_light(f"wh_flood_w{i}", (-115.0, y, 1.0), (-42.0, y * 0.6, 44.0), power=36000, kelvin=3100, size_deg=55, blend=0.55, radius=0.6, col=col)
        # a cooler wash on the north face so it is not black from behind
        C.spot_light("wh_flood_n", (10.0, 82.0, 1.0), (5.0, 22.0, 40.0), power=26000, kelvin=3800, size_deg=60, blend=0.6, radius=0.6, col=col)
    # interior atrium glow
    for i, z in enumerate((14.0, 34.0, 54.0)):
        for x in (-LENGTH / 2 + 8.0, LENGTH / 2 - 8.0):
            C.point_light(f"wh_atrium_l{i}_{'e' if x > 0 else 'w'}", (x, 0.0, z), power=3000, kelvin=3000, radius=2.0, col=col)
    return north
