"""Sky, haze, ground fog and moon.

`build_world` supports four sky modes:

* ``twilight`` (default) -- physically based Nishita sky with the sun a few
  degrees below the horizon, **plus** the real star field from
  ``tools/make_sky_hdri.py`` added on top. The twilight sky supplies the
  ambient light that makes the landscape readable; the stars survive only
  where it is dark, which is what actually happens during twilight.
* ``milkyway`` -- the star-field HDRI alone: full astronomical night.
* ``nishita`` -- the twilight sky alone, no stars.
* ``hdri`` -- a Poly Haven sky HDRI, rotated by ``rotation_deg``.

The star-field HDRI is already oriented in the local horizon frame, so it is
never rotated; the sun elevation/azimuth should come from the same instant
that generated it (``build_scene`` reads them from its JSON sidecar).

Aerial haze and ground fog are bounded box volumes, not World volumes: Cycles
treats a World volume as infinite, which attenuates the background and every
sun lamp to nothing.
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


def _nishita_sky(nt, sun_elevation_deg, sun_azimuth_deg):
    """Physically based sky. `sun_azimuth_deg` is a compass bearing (0 = north)."""
    sk = nt.nodes.new("ShaderNodeTexSky")
    sk.sky_type = "MULTIPLE_SCATTERING"
    sk.sun_disc = sun_elevation_deg > 0.0     # below the horizon there is no disc to draw
    sk.sun_elevation = math.radians(sun_elevation_deg)
    sk.sun_rotation = math.radians(sun_azimuth_deg) + NISHITA_ROT_OFFSET
    sk.altitude = 220.0                       # Batavia IL is ~220 m above sea level
    sk.air_density = 1.0
    sk.ozone_density = 1.6                    # more ozone deepens the twilight blue
    return sk.outputs["Color"]


def _env_sky(nt, path, vector_socket):
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(path, check_existing=True)
    env.interpolation = "Cubic"
    nt.links.new(vector_socket, env.inputs["Vector"])
    return env.outputs["Color"]


def build_world(
    sky_path: str,
    *,
    mode="twilight",
    strength=1.0,
    rotation_deg=0.0,
    sun_elevation_deg=-6.8,
    sun_azimuth_deg=283.0,
    star_scale=1.0,
    stars=False,
    star_strength=6.0,
):
    """Build the World shader. See the module docstring for the modes."""
    world = bpy.data.worlds.new("sky")
    world.use_nodes = True
    bpy.context.scene.world = world
    nt = world.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    # the star map is pre-oriented, so only a Poly Haven HDRI is ever rotated
    mp.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(rotation_deg if mode == "hdri" else 0.0))
    nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])

    have_map = bool(sky_path) and os.path.exists(sky_path)
    if mode in ("twilight", "milkyway") and not have_map:
        print(f"[atmosphere] star map missing ({sky_path}); falling back to nishita")
        mode = "nishita"

    if mode == "milkyway":
        sky = _env_sky(nt, sky_path, mp.outputs["Vector"])
        stars = False                                  # the map has real stars
    elif mode == "nishita":
        sky = _nishita_sky(nt, sun_elevation_deg, sun_azimuth_deg)
    elif mode == "hdri":
        sky = _env_sky(nt, sky_path, mp.outputs["Vector"]) if have_map else _nishita_sky(nt, sun_elevation_deg, sun_azimuth_deg)
    else:  # twilight: physical sky + real stars, added (radiance adds)
        twi = _nishita_sky(nt, sun_elevation_deg, sun_azimuth_deg)
        starmap = _env_sky(nt, sky_path, mp.outputs["Vector"])
        if star_scale != 1.0:
            _, starmap = C.mix_color(nt, 1.0, starmap, (star_scale, star_scale, star_scale, 1.0), "MULTIPLY")
        _, sky = C.mix_color(nt, 1.0, twi, starmap, "ADD")
        stars = False                                  # ditto
    sky_socket = sky

    # sky * strength
    _, sky_scaled = C.mix_color(nt, 1.0, sky_socket, (strength, strength, strength, 1.0), "MULTIPLY")
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


def build_haze(col, *, density=1.4e-4, size=200000.0, height=1800.0, anisotropy=0.35,
               scale_height=520.0, patch_scale=7000.0, patch=0.35, seed=7.0):
    """Aerial haze: exponential in height, patchy in plan, in a huge shallow box.

    The box has to cover the ground it is meant to fade. At 80 km it stopped
    well short of the far apron, which reaches the 167 km horizon, leaving the
    most distant ground unhazed and therefore too crisp exactly where the eye
    expects the horizon to dissolve.

    `density` is the density at the ground and it falls off as
    exp(-z / scale_height), where before it was uniform through the whole
    1800 m slab.

    What that buys, measured rather than reasoned about. Three renders of this
    frame -- no haze, the old uniform 3e-5, and this at 8e-5 -- give these mean
    luminances against the no-haze case:

                        uniform 3e-5     falloff 8e-5
        sky                  -0.00            -0.00
        horizon              -8.86            -8.88
        far ground           -7.29            -8.13
        middle               -0.91            -1.53
        near                 +1.44            +1.37

    So the haze was already doing the bulk of the work -- nine luminance units
    at the horizon and nothing at all to the sky -- and this change moved the
    horizon not at all and the far ground by 11 %. An earlier version of this
    docstring worked the sightline integrals and concluded the distance would
    roughly double its haze; the integrals were right about extinction and
    wrong about the result, because in-scattered skylight and city light rise
    with density too and largely cancel the extra extinction in the mean. Haze
    of this kind shows up as lost *contrast* in the distance, not as a darker
    distance.

    What the falloff does earn is the distribution. The sky column reads -0.00
    at both densities: rays leaving upward from a camera at 2600 m see 0.7 % of
    the ground density, so the star field is untouched however hazy the horizon
    gets, and the near ground (+1.4) stays crisp. A uniform slab cannot
    separate those -- it fades everything in proportion to path length alone --
    and it is what lets `density` be raised for a hazier distance without
    washing the stars out.

    The plan-view noise is the other half. A homogeneous volume gives the
    distance an even wash, and real air over a city at night is banded and
    uneven; a large-scale (7 km) modulation of +/- 35 % is what reads as
    atmosphere rather than as a fog filter.
    """
    mat = bpy.data.materials.new("aerial_haze")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.remove(nt.nodes["Principled BSDF"])
    out = nt.nodes["Material Output"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])   # object at z=-1: 1 m of offset

    # exp(-z / H), clamped at the ground so the box's buried floor does not
    # integrate a runaway density under the terrain
    zc = C.nmath(nt, "MAXIMUM", sep.outputs["Z"], value_b=0.0)
    zn = C.nmath(nt, "DIVIDE", zc.outputs[0], value_b=-scale_height)
    ex = C.nmath(nt, "EXPONENT", zn.outputs[0])

    n1 = nt.nodes.new("ShaderNodeTexNoise")
    n1.inputs["Scale"].default_value = 1.0 / patch_scale
    n1.inputs["Detail"].default_value = 2.0
    n1.inputs["Roughness"].default_value = 0.5
    n1.noise_dimensions = "4D"
    n1.inputs["W"].default_value = seed
    nt.links.new(tc.outputs["Object"], n1.inputs["Vector"])
    # Fac in [0,1] -> [1-patch, 1+patch]
    band = C.nmath(nt, "MULTIPLY_ADD", n1.outputs["Fac"], value_b=2.0 * patch)
    band.inputs[2].default_value = 1.0 - patch

    d = C.nmath(nt, "MULTIPLY", ex.outputs[0], band.outputs[0])
    d = C.nmath(nt, "MULTIPLY", d.outputs[0], value_b=density)

    vol = nt.nodes.new("ShaderNodeVolumePrincipled")
    nt.links.new(d.outputs[0], vol.inputs["Density"])
    vol.inputs["Anisotropy"].default_value = anisotropy
    vol.inputs["Color"].default_value = (0.72, 0.82, 1.0, 1.0)
    nt.links.new(vol.outputs["Volume"], out.inputs["Volume"])
    mat.cycles.volume_sampling = "EQUIANGULAR"
    bm = C.box_bmesh(size, size, height)
    obj = C.bmesh_object("aerial_haze", bm, col=col, material=mat, location=(0.0, 0.0, -1.0))
    obj.visible_shadow = False
    return obj


def build_ground_fog(col, *, center=(900.0, -450.0), size=(9000.0, 9000.0), height=170.0, density=3.2e-3, scale_height=26.0, patch_scale=300.0, anisotropy=0.62, seed=3.0):
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
