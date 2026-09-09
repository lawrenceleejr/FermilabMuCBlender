"""Sky, moon, haze and ground fog.

* World: Poly Haven blue-hour HDRI (dimmed, rotated) + procedural star field
  that only shows where the sky is dark, + a thin homogeneous haze volume for
  aerial perspective.
* Ground fog: a large box volume whose density decays exponentially with
  height and is broken into pools by low-frequency noise. Forward scattering
  (anisotropy) makes the collider beam and lamps bloom in the mist.
* Moon: a visible disc far away plus a matching sun lamp for silver
  backlight on fog and water.
"""
from __future__ import annotations

import math
import os

import bpy

from . import common as C

# Sky Texture `sun_rotation` was probed in Blender 5.2: 0 puts the sun at +Y
# (north) and 90 deg at +X (east), i.e. it already is a compass azimuth
# (clockwise from north), so no offset is needed.
NISHITA_ROT_OFFSET = 0.0


def build_world(hdri_path: str, *, mode="milkyway", strength=1.0, rotation_deg=0.0, sun_elevation_deg=-4.0, sun_azimuth_deg=290.0, stars=True, star_strength=6.0):
    """World shader. mode: "milkyway" (HDRI already in the local horizon frame,
    from tools/make_sky_hdri.py), "hdri" (Poly Haven sky, rotated), or
    "nishita" (physically based twilight)."""
    world = bpy.data.worlds.new("dusk")
    world.use_nodes = True
    bpy.context.scene.world = world
    nt = world.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(rotation_deg))
    nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])

    if mode == "milkyway" and os.path.exists(hdri_path):
        env = nt.nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(hdri_path, check_existing=True)
        env.interpolation = "Cubic"
        nt.links.new(mp.outputs["Vector"], env.inputs["Vector"])
        sky = env.outputs["Color"]
        stars = False  # real stars are in the map
    elif mode == "nishita" or not os.path.exists(hdri_path):
        # physically based twilight: sun just below the horizon -> deep blue
        # gradient with a warm band towards the sunset azimuth, no cloud deck
        sk = nt.nodes.new("ShaderNodeTexSky")
        sk.sky_type = "MULTIPLE_SCATTERING"
        sk.sun_disc = False
        sk.sun_elevation = math.radians(sun_elevation_deg)
        sk.sun_rotation = math.radians(sun_azimuth_deg) + NISHITA_ROT_OFFSET
        sk.altitude = 120.0
        sk.air_density = 1.0
        sk.ozone_density = 1.6      # more ozone = deeper blue twilight
        sky = sk.outputs["Color"]
    elif os.path.exists(hdri_path):
        env = nt.nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(hdri_path, check_existing=True)
        env.interpolation = "Cubic"
        nt.links.new(mp.outputs["Vector"], env.inputs["Vector"])
        sky = env.outputs["Color"]

    # sky * strength
    _, sky_scaled = C.mix_color(nt, 1.0, sky, (strength, strength, strength, 1.0), "MULTIPLY")
    color = sky_scaled

    if stars:
        # snap the view direction to a fine angular grid, hash -> sparse bright cells
        snap = nt.nodes.new("ShaderNodeVectorMath")
        snap.operation = "SNAP"
        snap.inputs[1].default_value = (0.0009, 0.0009, 0.0009)
        nt.links.new(tc.outputs["Generated"], snap.inputs[0])
        wn = nt.nodes.new("ShaderNodeTexWhiteNoise")
        wn.noise_dimensions = "3D"
        nt.links.new(snap.outputs["Vector"], wn.inputs["Vector"])
        thr = C.nmath(nt, "SUBTRACT", wn.outputs["Value"], value_b=0.9982)
        thr = C.nmath(nt, "MULTIPLY", thr.outputs[0], value_b=1.0 / 0.0018, clamp=True)
        thr = C.nmath(nt, "POWER", thr.outputs[0], value_b=2.5)
        # only above the horizon, and only where the sky itself is dark
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(tc.outputs["Generated"], sep.inputs["Vector"])
        up = C.nmath(nt, "GREATER_THAN", sep.outputs["Z"], value_b=0.03)
        lum = nt.nodes.new("ShaderNodeRGBToBW")
        nt.links.new(sky_scaled, lum.inputs["Color"])
        dark = C.nmath(nt, "MULTIPLY", lum.outputs["Val"], value_b=25.0, clamp=True)
        dark = C.nmath(nt, "SUBTRACT", value_a=1.0, b=dark.outputs[0])
        s = C.nmath(nt, "MULTIPLY", thr.outputs[0], up.outputs[0])
        s = C.nmath(nt, "MULTIPLY", s.outputs[0], dark.outputs[0])
        s = C.nmath(nt, "MULTIPLY", s.outputs[0], value_b=star_strength)
        # slight colour variety (blue-white .. warm)
        tint_k = C.nmath(nt, "MULTIPLY_ADD", wn.outputs["Value"], value_b=4000.0)
        tint_k.inputs[2].default_value = 4500.0
        star_col = C.blackbody(nt, tint_k.outputs[0])
        _, star_rgb = C.mix_color(nt, 1.0, star_col, (1, 1, 1, 1), "MULTIPLY")
        star_v = nt.nodes.new("ShaderNodeVectorMath")
        star_v.operation = "SCALE"
        nt.links.new(star_rgb, star_v.inputs[0])
        nt.links.new(s.outputs[0], star_v.inputs["Scale"])
        _, color = C.mix_color(nt, 1.0, sky_scaled, star_v.outputs["Vector"], "ADD")

    nt.links.new(color, bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 1.0
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

    # NOTE: no World volume -- Cycles treats it as infinite, which attenuates the
    # background and sun lamps to zero. Aerial haze lives in build_haze() instead.
    return world


def build_haze(col, *, density=3.5e-5, size=80000.0, height=1800.0, anisotropy=0.35):
    """Homogeneous haze in a huge shallow box: distant ground fades into the sky,
    while rays leaving through the top still see the sky and stars."""
    mat = bpy.data.materials.new("aerial_haze")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.remove(nt.nodes["Principled BSDF"])
    out = nt.nodes["Material Output"]
    vol = nt.nodes.new("ShaderNodeVolumePrincipled")
    vol.inputs["Density"].default_value = density
    vol.inputs["Anisotropy"].default_value = anisotropy
    vol.inputs["Color"].default_value = (0.72, 0.82, 1.0, 1.0)
    nt.links.new(vol.outputs["Volume"], out.inputs["Volume"])
    mat.cycles.volume_sampling = "EQUIANGULAR"
    bm = C.box_bmesh(size, size, height)
    obj = C.bmesh_object("aerial_haze", bm, col=col, material=mat, location=(0.0, 0.0, -1.0))
    obj.visible_shadow = False
    return obj


def build_ground_fog(col, *, center=(900.0, -450.0), size=(9000.0, 9000.0), height=140.0, density=1.2e-3, scale_height=18.0, patch_scale=420.0, anisotropy=0.55, seed=3.0):
    """Box volume: exponential height falloff x noise pools. Camera may be inside."""
    mat = bpy.data.materials.new("ground_fog")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.remove(nt.nodes["Principled BSDF"])
    out = nt.nodes["Material Output"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])  # object at z=0 => Object z == world z

    # exp(-z / H)
    zn = C.nmath(nt, "DIVIDE", sep.outputs["Z"], value_b=-scale_height)
    ex = C.nmath(nt, "EXPONENT", zn.outputs[0])
    # pools: two octaves of noise remapped to [0.15, 1.6]
    n1 = nt.nodes.new("ShaderNodeTexNoise")
    n1.inputs["Scale"].default_value = 1.0 / patch_scale
    n1.inputs["Detail"].default_value = 4.0
    n1.inputs["Roughness"].default_value = 0.55
    n1.noise_dimensions = "4D"
    n1.inputs["W"].default_value = seed
    nt.links.new(tc.outputs["Object"], n1.inputs["Vector"])
    pool = C.nmath(nt, "MULTIPLY_ADD", n1.outputs["Fac"], value_b=2.4)
    pool.inputs[2].default_value = -0.45
    pool = C.nmath(nt, "MAXIMUM", pool.outputs[0], value_b=0.12)
    n2 = nt.nodes.new("ShaderNodeTexNoise")
    n2.inputs["Scale"].default_value = 1.0 / (patch_scale * 6.0)
    n2.inputs["Detail"].default_value = 2.0
    nt.links.new(tc.outputs["Object"], n2.inputs["Vector"])
    big = C.nmath(nt, "MULTIPLY_ADD", n2.outputs["Fac"], value_b=1.4)
    big.inputs[2].default_value = 0.3
    d = C.nmath(nt, "MULTIPLY", ex.outputs[0], pool.outputs[0])
    d = C.nmath(nt, "MULTIPLY", d.outputs[0], big.outputs[0])
    d = C.nmath(nt, "MULTIPLY", d.outputs[0], value_b=density)

    vol = nt.nodes.new("ShaderNodeVolumePrincipled")
    nt.links.new(d.outputs[0], vol.inputs["Density"])
    vol.inputs["Anisotropy"].default_value = anisotropy
    vol.inputs["Color"].default_value = (0.86, 0.92, 1.0, 1.0)
    nt.links.new(vol.outputs["Volume"], out.inputs["Volume"])
    mat.cycles.volume_sampling = "MULTIPLE_IMPORTANCE"
    mat.cycles.volume_interpolation = "LINEAR"
    mat.cycles.volume_step_rate = 1.0

    bm = C.box_bmesh(size[0], size[1], height)
    obj = C.bmesh_object("ground_fog", bm, col=col, material=mat, location=(center[0], center[1], -0.5))
    obj.visible_shadow = False   # fog should not cast hard shadows onto the ground
    obj.visible_glossy = True
    return obj


def add_moon(col, *, azimuth_deg=62.0, elevation_deg=9.0, distance=32000.0, energy=0.45, kelvin=4300, disc_strength=15.0, angular_deg=0.9):
    """Moon: emissive disc at `distance` (angular size `angular_deg`; the real moon
    is 0.53 deg, a touch larger reads better on a cover) and a matching sun lamp."""
    d = C.direction_from(azimuth_deg, elevation_deg)
    r = math.tan(math.radians(angular_deg / 2)) * distance
    mat = C.emissive_material("moon", C.kelvin_rgb(4200), disc_strength)
    disc = C.disk_mesh("moon_disc", (0.0, 0.0), r, z=0.0, n=64, col=col, material=mat)
    disc.location = d * distance
    C.aim(disc, (0.0, 0.0, 0.0))
    disc.rotation_euler.rotate_axis("X", math.pi)  # face the scene
    disc.visible_shadow = False
    sun = C.sun_light("moonlight", azimuth_deg, elevation_deg, energy=energy, kelvin=kelvin, col=col)
    return disc, sun
