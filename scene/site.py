"""The Fermilab site: prairie terrain, the Tevatron and Main Injector berms
with their cooling ponds, lakes, roads and street lights, ancillary
buildings, tree cover, distant town glow -- and the proposed 10 km muon
collider ring drawn as a luminous beamline with its two detector halls.

Coordinates are metres, x east, y north, z up, origin at Wilson Hall.

The site's own geometry -- boundary, water, woods, roads, terrain and the
existing rings -- is read from the baked OpenStreetMap and DEM plan through
scene/geo.py rather than hand-placed. Only the proposed machines are authored
here, and their sizes come from the IMCC parameter list.
"""
from __future__ import annotations

import math
import random

import bpy

from . import common as C
from . import geo

# --- site plan: measured, not guessed ---------------------------------------
# Everything below the accelerator chain comes from `assets/plan/site_plan.json`
# (OpenStreetMap, ODbL) via scene/geo.py. It used to be hand-placed, and it was
# wrong in ways that mattered: the frame's origin sat 701 m south of Wilson
# Hall, the site polygon enclosed 20.7 km2 against a 27 km2 site, and the Main
# Injector was drawn about 1.2 km from where it is.
_TEV = geo.derived("tevatron")
_MI = geo.derived("main_injector")
_FILL = geo.derived("site_filler")

TEV_C, TEV_R = tuple(_TEV["center"]), _TEV["radius"]          # (835, -735), r 1000 m
MI_C = tuple(_MI["center"])                                   # (-533, -1154)
MI_RX = MI_RY = _MI["radius"]                                 # r 528 m from 3319 m
CAMPUS_BOUNDARY = [(p[0], p[1]) for p in geo.boundary()]
CAMPUS_AREA_KM2 = geo.boundary_area_km2()

# --- the muon-collider accelerator chain -------------------------------------
# Ring sizes are taken from the IMCC "Tentative Parameter list for the
# International Muon Collider Collaboration", 30 October 2023 (indico.cern.ch
# event 1313021), Table 3.10 for the acceleration chain, Table 3.19 for the
# collider and Table 3.2 for the proton-driver compressor. Radii are
# circumference / 2 pi.
#
#   collider      C = 10 000 m  -> r = 1591.5 m   (10 TeV; the 3 TeV option is 4.5 km)
#   RCS1, RCS2    C =  5 990 m  -> r =  953.3 m   (one tunnel, two machines)
#   RCS3          C = 10 700 m  -> r = 1702.7 m
#   compressor    C = 300-900 m -> r =   48-143 m
#
# RCS 4 is the exception, deliberately. The IMCC reference is 35 000 m, which
# is 11.1 km across and does not fit a 5.7 x 6.2 km site -- drawn at that size
# it left the property on all four sides, and the figure's strongest line was
# the one contradicting its own claim. This design instead sizes the final
# synchrotron to the site: a *site filler*, the largest ring the campus can
# hold. geo measures that as the largest circle inscribed in the real boundary,
# r 2558 m, less a 250 m setback from the property line -> r 2308 m,
# circumference 14.50 km. It is a constraint read off the site rather than a
# number chosen to fit the picture, and a 14.5 km final stage needs more
# acceleration turns than the 35 km reference -- a trade the label states.
TAU = 2.0 * math.pi
MC_R = 10000.0 / TAU                         # 1591.5 m
RCS12_R = 5990.0 / TAU                       # 953.3 m -- 95 % of the Tevatron's 6283 m
RCS3_R = 10700.0 / TAU                       # 1702.7 m
RCS4_R = _FILL["radius"]                     # 2307.9 m: the site filler
RCS4_C = tuple(_FILL["center"])              # (1522, 387), the inscribed centre
RCS4_CIRC_M = _FILL["circumference_m"]       # 14 501 m
IMCC_RCS4_C_M = 35000.0                      # the reference the site filler replaces

PD_ACCUM_R = 900.0 / TAU                     # 143.2 m, compressor upper option
PD_BUNCH_R = 600.0 / TAU                     # 95.5 m, mid-range

# The chain is laid out around the site-filler centre, so the collider sits in
# the middle of the campus instead of off toward one edge.
MC_C = RCS4_C
MC_IP_ANGLES = (120.0, 300.0)                # detector halls, diametrically opposite

# The proton driver and cooling channel run in from the south-west, the part of
# the campus with no existing machine in it.
PD_ACCUM_C = (-560.0, -2180.0)
PD_BUNCH_C = (-560.0, -2430.0)
TARGET_XY = (-330.0, -1960.0)                # pion production target hall
LINAC = ((-1750.0, -2180.0), (-720.0, -2180.0))

# Ionisation cooling: the document specifies 10 "B-type" rectilinear stages
# S1-S10 plus A-stages, bunch merge and final cooling, but no overall length,
# so the channel is drawn at an indicative 600 m and labelled by stage count.
COOLING_PATH = [(-250.0, -1870.0), (-60.0, -1760.0), (140.0, -1650.0),
                (330.0, -1540.0), (500.0, -1430.0)]
COOLING_MODULES = 20

# RCS 1 and 2 share the Tevatron tunnel, so they are drawn on the Tevatron's
# own circle rather than as a separate ring 47 m inside it: claiming reuse
# while drawing two distinct tunnels was an inconsistency a reader could
# measure. The 5990 m machine occupies 95 % of the 6283 m tunnel.
RCS12_C = TEV_C
RCS12_DRAW_R = TEV_R - 30.0                  # a drawing offset, not a second tunnel
RCS3_C = RCS4_C
RCS_C = TEV_C
RCS_RADII = (RCS12_DRAW_R, RCS3_R)

SITE_X = (min(p[0] for p in CAMPUS_BOUNDARY), max(p[0] for p in CAMPUS_BOUNDARY))
SITE_Y = (min(p[1] for p in CAMPUS_BOUNDARY), max(p[1] for p in CAMPUS_BOUNDARY))

COLLIDER_COLOR = (0.16, 0.62, 1.0)
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

    # Agricultural parcels. The land around and inside the site is a patchwork of
    # mown grass, restored prairie and leased farm fields; from the air that
    # parcel structure is most of what makes the landscape readable, so give each
    # Voronoi cell its own brightness and warmth. Subtle -- +-30 % -- so it reads
    # as fields rather than as a texture.
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.feature = "F1"
    vor.distance = "EUCLIDEAN"
    vor.inputs["Scale"].default_value = 1.0 / 340.0     # ~340 m parcels
    vor.inputs["Randomness"].default_value = 0.85
    nt.links.new(tc.outputs["Object"], vor.inputs["Vector"])
    cell = nt.nodes.new("ShaderNodeTexWhiteNoise")
    cell.noise_dimensions = "3D"
    nt.links.new(vor.outputs["Position"], cell.inputs["Vector"])
    parcel = C.nmath(nt, "MULTIPLY_ADD", cell.outputs["Value"], value_b=0.80)
    parcel.inputs[2].default_value = 0.60              # 0.60 .. 1.40
    _, parcelled = C.mix_color(nt, 1.0, mixed, parcel.outputs[0], "MULTIPLY")

    # ...and its own hue: September tallgrass and stubble run tan/gold, mown grass
    # and soybean parcels stay green, so mix the two per cell.
    cell_hue = nt.nodes.new("ShaderNodeTexWhiteNoise")
    cell_hue.noise_dimensions = "4D"
    cell_hue.inputs["W"].default_value = 3.7
    nt.links.new(vor.outputs["Position"], cell_hue.inputs["Vector"])
    _, field_tint = C.mix_color(nt, cell_hue.outputs["Value"], (0.84, 0.68, 0.40, 1.0), (0.50, 0.66, 0.38, 1.0))
    _, tinted = C.mix_color(nt, 1.0, parcelled, field_tint, "MULTIPLY")
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


TERRAIN_GRID = 384          # 60 km / 384 = 156 m per quad
TERRAIN_HALF = 30000.0


def build_terrain(col, mat, *, grid=TERRAIN_GRID):
    """The ground, displaced by a real elevation model.

    Northern Illinois is glacial plain, so there was a temptation to invent
    hills for the distance view. The DEM says not to: within 6 km of the site
    the relief is 85 m, and the campus itself is flat to about a tenth of a
    metre. What the horizon actually has is long-wavelength moraine relief --
    267 m across the 60 km box, rising to the north-west -- which is invisible
    at close range and shapes the whole skyline at 20 km. Inventing mountains
    would have replaced the one thing the terrain does contribute.

    Falls back to a flat quad when the baked DEM is absent, so the scene still
    builds from a bare checkout.
    """
    if not geo.has_dem():
        print("[site] no DEM in the baked plan; terrain is flat")
        C.mesh_object("terrain",
                      [(-TERRAIN_HALF, -TERRAIN_HALF, 0), (TERRAIN_HALF, -TERRAIN_HALF, 0),
                       (TERRAIN_HALF, TERRAIN_HALF, 0), (-TERRAIN_HALF, TERRAIN_HALF, 0)],
                      [(0, 1, 2, 3)], col=col, material=mat)
        return
    verts, faces = geo.dem_grid(grid, TERRAIN_HALF)
    zs = [v[2] for v in verts]
    print(f"[site] terrain {grid}x{grid} over {2 * TERRAIN_HALF / 1000:.0f} km, "
          f"z {min(zs):+.0f}..{max(zs):+.0f} m")
    C.mesh_object("terrain", verts, faces, col=col, material=mat, smooth=True)


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
    # Faint amber markers along the existing machines' crests, so the reused
    # infrastructure reads at overview scale and can carry a label. Without
    # these the berms are unlit prairie and effectively invisible at night.
    tev_line = C.emissive_material("tevatron_tunnel_glow", (1.0, 0.62, 0.30), 0.5, camera_strength=1.6)
    C.tube_mesh("tevatron_tunnel_line", C.circle_points(TEV_C, TEV_R, n=480, z=6.6), 0.6, sides=6, col=col, material=tev_line, closed=True)
    mi_line = C.emissive_material("main_injector_glow", (1.0, 0.58, 0.26), 0.35, camera_strength=1.1)
    C.tube_mesh("main_injector_line", C.circle_points(MI_C, MI_RX, n=360, ry=MI_RY, z=6.6), 0.6, sides=6, col=col, material=mi_line, closed=True)


def build_collider(col, concrete):
    """The proposed muon collider: 10 km luminous ring, two detector halls,
    and the proton-driver linac feeding the western crossing."""
    core = C.emissive_material("collider_beam", COLLIDER_COLOR, 8.0, camera_strength=38.0)
    C.tube_mesh("muon_collider_ring", C.circle_points(MC_C, MC_R, n=900, z=2.2), 1.0, sides=10, col=col, material=core, closed=True)
    # soft outer sheath (dimmer, wider) gives the line body without blowing out
    sheath = C.emissive_material("collider_sheath", (0.35, 0.75, 1.0), 0.5, camera_strength=2.0)
    C.tube_mesh("muon_collider_sheath", C.circle_points(MC_C, MC_R, n=900, z=2.2), 2.3, sides=10, col=col, material=sheath, closed=True)

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

    _build_rcs(col, concrete)
    _build_proton_driver(col, concrete)
    _build_cooling_channel(col, concrete)

    linac = C.emissive_material("linac_beam", (0.85, 0.95, 1.0), 1.5, camera_strength=9.0)
    (x0, y0), (x1, y1) = LINAC
    # 2.2 m radius: at ~8 km a thinner tube falls below a pixel and the label
    # would point at nothing. Emission plus bloom carries it from there.
    C.tube_mesh("proton_driver_linac", [(x0, y0, 2.2), (x1, y1, 2.2)], 2.2, sides=8, col=col, material=linac)
    C.box_object("target_hall", (40.0, 18.0, 10.0), (x0 + 0.62 * (x1 - x0), y0 + 22.0, 0), col=col, material=concrete)


def build_campus_boundary(col, *, width=11.0, strength=1.2, camera_strength=4.0, color=(1.0, 0.93, 0.82), z=0.7, posts=True):
    """Highlight the site boundary as a glowing ribbon lying on the ground.

    A flat ribbon (rather than a tube) reads as a clean line from an overhead
    or overview camera, and avoids drawing a glowing pipe across the close
    shots. It does *not* vanish from a low camera though -- seen edge-on it
    becomes a bright streak along the horizon -- so build_scene only builds it
    for the cameras in BOUNDARY_CAMERAS by default. Kept dimmer than the
    collider so the visual hierarchy stays collider > Tevatron > boundary.
    """
    mat = C.emissive_material("campus_boundary", color, strength, camera_strength=camera_strength)
    pts = CAMPUS_BOUNDARY
    obj = C.ribbon_mesh("campus_boundary", pts, width, z, col=col, material=mat, closed=True)
    obj.visible_shadow = False
    obj.visible_diffuse = False        # a boundary marker should not light the prairie
    obj.visible_glossy = False
    obj.visible_volume_scatter = False
    if posts:
        # low markers at the vertices so the corners register at a distance
        post_mat = C.emissive_material("campus_post", color, strength * 1.5, camera_strength=camera_strength * 2.0)
        me = C.sphere_mesh_data("campus_post_m", 9.0, subdiv=1)
        me.materials.append(post_mat)
        for i, (x, y) in enumerate(pts):
            o = C.instance(f"campus_post_{i}", me, col=col, location=(x, y, z + 4.0))
            o.visible_shadow = False
            o.visible_diffuse = False
            o.visible_glossy = False
            o.visible_volume_scatter = False
    return obj


# (name, centre, drawn radius) for the synchrotron chain. RCS 1 and 2 ride the
# Tevatron tunnel; RCS 3 and the site filler are centred on the campus.
RCS_RINGS = (
    ("rcs12", RCS12_C, RCS12_DRAW_R),
    ("rcs3", RCS3_C, RCS3_R),
    ("rcs4", RCS4_C, RCS4_R),
)


def _build_rcs(col, concrete):
    """The rapid-cycling synchrotron tunnels.

    RCS 1 and 2 share one 5 990 m tunnel and ride the existing Tevatron ring,
    so they are drawn on the Tevatron's own circle -- offset 30 m purely so
    both lines are visible -- rather than as a separate ring 47 m inside it.
    Claiming tunnel reuse while drawing two distinct tunnels whose
    circumferences differ by 5 % was an inconsistency a reader could measure
    off the scale bar.

    RCS 4 is the site filler: at 14.5 km it is the largest ring the campus can
    hold, and it fits, where the IMCC's 35 km reference is 11 km across and
    left the property on all four sides. It is now the outermost proposed ring
    rather than a curve sweeping off the edge of the frame.
    """
    tunnel = C.emissive_material("rcs_tunnel", (0.42, 0.78, 1.0), 1.2, camera_strength=7.0)
    for name, c, r in RCS_RINGS:
        C.tube_mesh(f"{name}_tunnel", C.circle_points(c, r, n=720, z=4.4), 0.7,
                    sides=6, col=col, material=tunnel, closed=True)
    # RF straights: short brighter runs so the rings read as machines, not contours
    rf = C.emissive_material("rcs_rf", (0.62, 0.88, 1.0), 2.5, camera_strength=16.0)
    for name, c, r in RCS_RINGS:
        for k in range(4):
            a0 = math.radians(90 * k + 12)
            a1 = a0 + math.radians(11)
            pts = [(c[0] + r * math.cos(a), c[1] + r * math.sin(a), 4.4)
                   for a in [a0 + (a1 - a0) * j / 8 for j in range(9)]]
            C.tube_mesh(f"{name}_rf_{k}", pts, 1.5, sides=6, col=col, material=rf)


def _build_proton_driver(col, concrete):
    """Proton driver complex: linac hall, accumulator and buncher rings.

    The linac itself is built in build_collider (it predates this function);
    here are the rings that bunch the beam before it hits the target.
    """
    ring = C.emissive_material("pd_ring", (0.80, 0.90, 1.0), 1.8, camera_strength=11.0)
    for name, c, r in (("accumulator", PD_ACCUM_C, PD_ACCUM_R), ("buncher", PD_BUNCH_C, PD_BUNCH_R)):
        C.tube_mesh(f"pd_{name}", C.circle_points(c, r, n=160, z=3.0), 1.1, sides=6,
                    col=col, material=ring, closed=True)
    # service halls along the complex
    unit = C.box_mesh_data("pd_box")
    hall = _lit_box_material("pd_hall", (0.26, 0.26, 0.25), lit_color=C.kelvin_rgb(4000),
                             lit_strength=0.7, band=(0.45, 0.72), lit_fraction=0.35, seed=17.0)
    for i, (x, y, sx, sy, sz) in enumerate((
        (-640.0, -1690.0, 90.0, 26.0, 9.0),
        (-380.0, -1610.0, 54.0, 24.0, 11.0),
        (-250.0, -1900.0, 40.0, 20.0, 8.0),
    )):
        me = unit.copy()
        me.materials.append(hall)
        C.instance(f"pd_hall_{i}", me, col=col, location=(x, y, 0.0), scale=(sx, sy, sz))
    C.point_light("pd_light", (-430.0, -1750.0, 30.0), power=14000, kelvin=4200, radius=8.0, col=col)


def _build_cooling_channel(col, concrete):
    """Muon ionisation cooling complex: a chain of absorber/RF modules.

    Modelled as discrete modules on a bright beamline, because that beading is
    what distinguishes a cooling channel from any other length of beam pipe.
    """
    beam = C.emissive_material("cooling_beam", (0.62, 0.95, 0.95), 2.0, camera_strength=13.0)
    mod = C.emissive_material("cooling_module", (0.85, 1.0, 1.0), 3.0, camera_strength=26.0)
    pts = [(x, y, 3.0) for x, y in COOLING_PATH]
    C.tube_mesh("cooling_beamline", pts, 1.2, sides=8, col=col, material=beam)

    # walk the polyline at even parameter and drop a module at each step
    me = C.sphere_mesh_data("cooling_module_m", 3.4, subdiv=1)
    me.materials.append(mod)
    segs = list(zip(COOLING_PATH, COOLING_PATH[1:]))
    for i in range(COOLING_MODULES):
        t = i / (COOLING_MODULES - 1) * len(segs)
        k = min(int(t), len(segs) - 1)
        f = t - k
        (x0, y0), (x1, y1) = segs[k]
        C.instance(f"cooling_module_{i}", me, col=col,
                   location=(x0 + (x1 - x0) * f, y0 + (y1 - y0) * f, 4.0))
    hall = C.box_object("cooling_hall", (70.0, 30.0, 12.0), (-350.0, -1560.0, 0.0),
                        col=col, material=concrete)
    C.point_light("cooling_light", (-100.0, -1300.0, 28.0), power=12000, kelvin=5200, radius=8.0, col=col)
    return hall


# --------------------------------------------------------------------------- #
# named features, for the annotation layer (tools/annotate.py)
# --------------------------------------------------------------------------- #
def _ring_pt(c, r, angle_deg, z=0.0, ry=None):
    a = math.radians(angle_deg)
    return (c[0] + r * math.cos(a), c[1] + (ry or r) * math.sin(a), z)


NNBSP = "\u202f"          # narrow no-break space: the SI/ISO 31-0 group separator


def circ_km(radius_m: float) -> str:
    """A ring's circumference as a km string, computed from the radius drawn.

    The labels used to carry hand-typed metres -- "10 000 m", "35 000 m" -- next
    to a km scale bar and a km2 area, which made the reader do the arithmetic
    the figure exists to save them, and left the numbers free to drift from the
    geometry. Deriving them here means a label cannot disagree with its ring.
    """
    return f"{2.0 * math.pi * radius_m / 1000.0:.3g}{NNBSP}km"


def ellipse_circ_km(rx: float, ry: float) -> str:
    """Ramanujan's approximation; good to ~1e-5 at these eccentricities."""
    h = ((rx - ry) / (rx + ry)) ** 2
    c = math.pi * (rx + ry) * (1.0 + 3.0 * h / (10.0 + math.sqrt(4.0 - 3.0 * h)))
    return f"{c / 1000.0:.3g}{NNBSP}km"


def polygon_area_km2(pts) -> float:
    """Shoelace area of a closed ground polygon, in km2."""
    a = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i][0], pts[i][1]
        x1, y1 = pts[(i + 1) % n][0], pts[(i + 1) % n][1]
        a += x0 * y1 - x1 * y0
    return abs(a) / 2.0 / 1e6


def annotation_anchors() -> dict[str, dict]:
    """What is worth labelling, with the copy and where the leader attaches.

    Three anchor kinds. `pos` is a fixed world point. `ring`/`path` give a curve
    to sample, and `prefer` a normalised screen target: build_scene projects every
    sample and attaches the leader to whichever lands nearest that target and is
    on screen. That keeps a label on a clear stretch of its own arc without
    hard-coding an angle that would silently go wrong if the camera moved.

    Only features that actually render at overview scale appear here -- a leader
    pointing at nothing is worse than no leader.
    """
    return {
        "collider": dict(
            ring=dict(center=MC_C, radius=MC_R, z=3.0),
            prefer=(0.78, 0.56),                       # east arc: keeps the collider in the right column
            label="Muon Collider Ring",
            metric=f"{circ_km(MC_R)} circumference",
            accent="collider",
        ),
        "detector_a": dict(
            pos=_mc_point(MC_IP_ANGLES[0], z=2.0),
            label="Detector Hall A",
            metric="interaction point 1",
            accent="collider",
        ),
        "detector_b": dict(
            pos=_mc_point(MC_IP_ANGLES[1], z=2.0),
            label="Detector Hall B",
            metric="interaction point 2",
            accent="collider",
        ),
        "proton_driver": dict(
            pos=(PD_ACCUM_C[0], PD_ACCUM_C[1] + PD_ACCUM_R, 3.0),
            label="Proton Driver Complex",
            metric="linac, accumulator, target",
            accent="collider",
        ),
        "cooling": dict(
            pos=(COOLING_PATH[-1][0], COOLING_PATH[-1][1], 4.0),
            label="Muon Cooling Complex",
            metric="ionisation cooling channel",
            accent="collider",
        ),
        "rcs12": dict(
            ring=dict(center=RCS12_C, radius=RCS12_DRAW_R, z=4.4),
            prefer=(0.30, 0.71),
            label="RCS 1 & 2",
            metric=f"in the Tevatron tunnel, {circ_km(RCS12_R)} of {circ_km(TEV_R)}",
            accent="collider",
        ),
        "rcs3": dict(
            ring=dict(center=RCS3_C, radius=RCS3_R, z=4.4),
            prefer=(0.74, 0.66),
            label="RCS 3",
            metric=circ_km(RCS3_R),
            accent="collider",
        ),
        "rcs4": dict(
            ring=dict(center=RCS4_C, radius=RCS4_R, z=4.4),
            prefer=(0.15, 0.50),
            label="RCS 4 site filler",
            metric=f"{circ_km(RCS4_R)}, largest ring the site holds",
            accent="collider",
        ),
        "tevatron": dict(
            ring=dict(center=TEV_C, radius=TEV_R, z=6.6),
            prefer=(0.24, 0.66),
            label="Tevatron Ring",
            metric=f"{circ_km(TEV_R)}, tunnel reused",
            accent="tevatron",
        ),
        "main_injector": dict(
            ring=dict(center=MI_C, radius=MI_RX, ry=MI_RY, z=6.6),
            prefer=(0.13, 0.73),
            label="Main Injector",
            metric=f"{ellipse_circ_km(MI_RX, MI_RY)}, existing",
            accent="tevatron",
        ),
        "wilson": dict(
            pos=(0.0, 0.0, 78.0),
            label="Wilson Hall",
            metric="central laboratory",
            accent="neutral",
        ),
        "boundary": dict(
            path=CAMPUS_BOUNDARY,
            z=2.0,
            prefer=(0.40, 0.80),                       # the near south edge, below the rings
            label="Fermilab Site",
            metric=f"{CAMPUS_AREA_KM2:.0f}{NNBSP}km\u00b2, {CAMPUS_AREA_KM2 * 247.105:.0f} acres",
            accent="boundary",
        ),
    }


# --------------------------------------------------------------------------- #
# water
# --------------------------------------------------------------------------- #
def build_water(col, water):
    """Every mapped water body inside 9 km, as its real outline.

    These used to be ten hand-placed ellipses. The Tevatron's cooling ponds
    follow the ring, the Main Injector's pond arcs around its berm, and the
    site's lakes have shapes a viewer at this scale can read -- none of which
    an ellipse at a guessed position reproduces.

    Each pond is laid flat at the ground height of its own centroid: sloping a
    lake with the terrain looks broken, and the campus is flat enough that one
    height per body is right to well under a metre.
    """
    n = 0
    for w in geo.ways("water", closed=True, min_pts=4, radius=9000.0):
        pts = w["pts"]
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        z = geo.elev(cx, cy) + 0.08
        ring = clean_way(pts, True)
        if len(ring) < 3:
            continue
        verts = [(p[0], p[1], z) for p in ring]
        C.mesh_object(f"water_{w['name'] or n}_{n}", verts, [tuple(range(len(verts)))],
                      col=col, material=water)
        n += 1
    print(f"[site] {n} mapped water bodies")
    return n


# --------------------------------------------------------------------------- #
# roads and lights
# --------------------------------------------------------------------------- #
# Which OSM road layers to draw, how wide, and how far out to bother.
ROAD_LAYERS = (
    ("major_roads", 11.0, 16000.0),
    ("minor_roads", 7.5, 7000.0),
    ("site_roads", 6.0, 5000.0),
)


def asphalt_material():
    return C.pbr_material("asphalt", "Asphalt031_1K-JPG", scale=6.0, tint=(0.55, 0.55, 0.58), roughness_mult=0.55, normal_strength=0.5, projection="FLAT")


def build_roads(col, asphalt):
    """The real road network, draped on the terrain.

    The hand-authored version was eleven straight lines on a grid. The mapped
    network is what gives the distance view its structure -- the section-line
    grid of northern Illinois, the diagonal of the rail corridor, the curve of
    the Tevatron's ring road -- and it is also what the street lights need,
    since lights belong on roads that exist.
    """
    n = 0
    for layer, width, radius in ROAD_LAYERS:
        for w in geo.ways(layer, min_pts=2, radius=radius):
            closed = bool(w.get("closed"))
            pts = clean_way(w["pts"], closed)
            if len(pts) < (3 if closed else 2):
                continue
            pts3 = [(x, y, geo.elev(x, y) + 0.12) for x, y in pts]
            C.ribbon_mesh(f"road_{layer}_{n}", pts3, width, None, col=col,
                          material=asphalt, closed=closed)
            n += 1
    print(f"[site] {n} mapped road segments")
    return n


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


def clean_way(pts, closed=False, *, eps=0.5):
    """Drop repeated vertices, and the duplicated closing vertex on a ring.

    OSM writes a closed way with its first point repeated at the end. Handing
    that to ribbon_mesh with closed=True makes it wrap onto a face that already
    exists, which bmesh refuses -- so the endpoint comes off here rather than at
    every call site.
    """
    out = []
    for p in pts:
        q = (p[0], p[1])
        if out and math.hypot(q[0] - out[-1][0], q[1] - out[-1][1]) < eps:
            continue
        out.append(q)
    if closed and len(out) > 1 and math.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) < eps:
        out.pop()
    return out


def _resample(pts, spacing):
    """Points every `spacing` metres along a polyline."""
    out, carry = [], 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg < 1e-6:
            continue
        t = carry
        while t < seg:
            f = t / seg
            out.append((x0 + (x1 - x0) * f, y0 + (y1 - y0) * f))
            t += spacing
        carry = t - seg
    return out


def build_street_lights(col, *, near=1800.0, far=12000.0, cap=4200):
    """Lights on the roads that exist, near ones as poles and far ones as points.

    "Street lights in the distance" is most of what tells a viewer this is an
    inhabited landscape rather than an empty plain, and at 10 km a lamp is a
    point of light -- a pole would be a tenth of a pixel. So the near field
    gets pole-and-head geometry and everything beyond `near` gets a bare
    emissive point at lamp height, which is both what it looks like and what
    keeps the instance count somewhere Cycles can sample.

    Capped, because the mapped network inside 12 km carries a few hundred
    kilometres of road and lighting all of it at 100 m spacing would be tens of
    thousands of emitters.
    """
    lamp = _lamp_factory(col)
    sodium = C.emissive_material("lamp_far_sodium", SODIUM, 26.0, camera_strength=30.0)
    head_far = C.sphere_mesh_data("lamp_far_head", 1.1)
    head_far.materials.append(sodium)
    rng = random.Random(41)
    n_near = n_far = 0

    for layer, spacing, radius in (("major_roads", 105.0, far),
                                   ("site_roads", 70.0, near)):
        for w in geo.ways(layer, min_pts=2, radius=radius):
            pts = [(p[0], p[1]) for p in w["pts"]]
            for i, (x, y) in enumerate(_resample(pts, spacing)):
                d = math.hypot(x, y)
                if d > radius:
                    continue
                # thin them out with distance: a far road reads as a dotted
                # line of light, not a continuous strip
                if d > near and rng.random() > 0.55:
                    continue
                if d <= near:
                    lamp(x, y + (6.0 if i % 2 else -6.0), "sodium")
                    n_near += 1
                elif n_far < cap:
                    C.instance(f"lamp_far_{n_far}", head_far, col=col,
                               location=(x, y, geo.elev(x, y) + 9.2))
                    n_far += 1
    print(f"[site] street lights: {n_near} poles inside {near / 1000:.1f} km, "
          f"{n_far} distant points out to {far / 1000:.0f} km")
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
        "glass": _lit_box_material("bld_glass", (0.06, 0.07, 0.08), lit_color=C.kelvin_rgb(4300), lit_strength=0.55, band=(0.3, 0.7), lit_fraction=0.35, seed=3.0),
        "office": _lit_box_material("bld_office", (0.30, 0.28, 0.25), lit_color=C.kelvin_rgb(3300), lit_strength=0.7, band=(0.4, 0.68), lit_fraction=0.3, seed=5.0),
        "industrial": _lit_box_material("bld_industrial", (0.22, 0.22, 0.21), lit_color=C.kelvin_rgb(3800), lit_strength=0.55, band=(0.58, 0.76), lit_fraction=0.16, seed=9.0),
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
    house = _lit_box_material("bld_house", (0.35, 0.30, 0.26), lit_color=C.kelvin_rgb(2900), lit_strength=0.7, band=(0.35, 0.65), lit_fraction=0.45, seed=11.0)
    for i in range(34):
        x = 2850.0 + rng.uniform(-260, 260)
        y = 250.0 + rng.uniform(-220, 220)
        me = unit.copy()
        me.materials.append(house)
        C.instance(f"village_{i}", me, col=col, location=(x, y, 0.0), rotation=(0, 0, rng.uniform(0, math.pi)), scale=(rng.uniform(10, 18), rng.uniform(9, 14), rng.uniform(5, 8)))


# --------------------------------------------------------------------------- #
# trees
# --------------------------------------------------------------------------- #
_WATER_RINGS = None


def _water_rings():
    global _WATER_RINGS
    if _WATER_RINGS is None:
        _WATER_RINGS = [[(p[0], p[1]) for p in w["pts"]]
                        for w in geo.ways("water", closed=True, min_pts=4, radius=9000.0)]
    return _WATER_RINGS


def _keep_out(x, y):
    """Avoid the machines, the water and Wilson Hall when scattering trees."""
    if -140 < x < 330 and -110 < y < 110:            # Wilson Hall + plaza + pond
        return False
    if dist_to_ring((x, y), TEV_C, TEV_R) < 60:
        return False
    if dist_to_ring((x, y), MI_C, MI_RX) < 55:
        return False
    if dist_to_ring((x, y), MC_C, MC_R) < 25:
        return False
    for _, c, r in RCS_RINGS:
        if dist_to_ring((x, y), c, r) < 22:
            return False
    for ring in _water_rings():
        if geo.point_in(ring, x, y):
            return False
    for _, (bx, by), (sx, sy, _), _, _ in BUILDINGS:
        if abs(x - bx) < sx / 2 + 8 and abs(y - by) < sy / 2 + 8:
            return False
    return True


def build_trees(col, *, density=1.0, spacing=15.0, cap=26000):
    """Trees where OpenStreetMap says there are trees.

    The nine hand-drawn ellipses this replaces were guesses at where Fermilab's
    woodland is. The mapped `natural=wood` and `landuse=forest` polygons put it
    where it is -- the Big Woods, the riparian strips along Indian Creek and the
    Fox River, and the blocks of forest preserve out to 14 km that give the
    distance view its treeline.

    Points come from a jittered grid inside each polygon rather than uniform
    random placement, which clumps badly enough at these counts to read as
    noise instead of canopy.
    """
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

    def tree(x, y, s=None, *, trunked=True):
        s = s or rng.uniform(5.5, 9.5)
        n[0] += 1
        z = geo.elev(x, y)
        C.instance(f"tree_{n[0]}", canopies[n[0] % 4], col=col,
                   location=(x, y, z + 4.2 + 0.55 * s), rotation=(0, 0, rng.uniform(0, 6.28)),
                   scale=(s, s * rng.uniform(0.85, 1.1), s * 0.9))
        if trunked:
            C.instance(f"trunk_{n[0]}", trunk, col=col, location=(x, y, z),
                       scale=(1, 1, 0.9 + 0.1 * s))

    polys = []
    for layer in ("wood", "forest"):
        for w in geo.ways(layer, closed=True, min_pts=4, radius=14000.0):
            polys.append([(p[0], p[1]) for p in w["pts"]])
    step = spacing / max(density, 0.05) ** 0.5
    pts = geo.scatter_in_polygons(polys, step, rng, limit=cap)
    for x, y in pts:
        if not _keep_out(x, y):
            continue
        far = math.hypot(x, y) > 3500.0
        tree(x, y, rng.uniform(6.0, 10.5), trunked=not far)
    # hedgerows along the mapped field roads: the prairie's other vertical
    for w in geo.ways("minor_roads", min_pts=2, radius=6000.0):
        line = [(p[0], p[1]) for p in w["pts"]]
        for x, y in _resample(line, 26.0):
            if rng.random() < 0.13 and _keep_out(x, y):
                tree(x + rng.uniform(-6, 6), y + rng.uniform(-6, 6), rng.uniform(5, 8))
    print(f"[site] {n[0]} trees from {len(polys)} mapped wood/forest polygons")
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
    """Distant town glow as emissive strips on the horizon.

    Off by default (--towns to enable): the light-pollution domes baked into the
    sky HDRI by tools/make_sky_hdri.py put that glow where it belongs, in the
    sky, whereas these strips sit 9-26 km out and only 30-70 m tall, so from an
    elevated camera they are silhouetted *against the ground* as a dark bar
    rather than glowing above the horizon.
    """
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
