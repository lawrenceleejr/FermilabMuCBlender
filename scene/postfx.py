"""Cycles settings, colour management and the compositor look.

Compositor (Blender 5.x node-group API): Render Layers -> Bloom glare ->
gentle chromatic aberration -> vignette -> Group Output.
"""
from __future__ import annotations

import bpy

from . import common as C


def configure_cycles(scene, *, samples, adaptive=True, time_limit=0, denoise=True, threads=0):
    scene.render.engine = "CYCLES"
    cy = scene.cycles
    cy.device = "CPU"
    cy.samples = samples
    cy.use_adaptive_sampling = adaptive
    cy.adaptive_threshold = 0.012
    cy.adaptive_min_samples = max(16, samples // 16)
    cy.time_limit = time_limit
    cy.use_denoising = denoise
    try:
        cy.denoiser = "OPENIMAGEDENOISE"
        cy.denoising_input_passes = "RGB_ALBEDO_NORMAL"
        cy.denoising_prefilter = "ACCURATE"
        cy.denoising_quality = "HIGH"
    except Exception as e:  # noqa: BLE001
        print("[postfx] denoiser config:", e)
    cy.use_light_tree = True
    cy.use_guiding = True             # path guiding: big win for fog + many small lights on CPU
    cy.use_surface_guiding = True
    cy.use_volume_guiding = True
    cy.max_bounces = 8
    cy.diffuse_bounces = 3
    cy.glossy_bounces = 4
    cy.transmission_bounces = 6
    cy.transparent_max_bounces = 8
    cy.volume_bounces = 1
    cy.volume_step_rate = 1.0
    cy.volume_max_steps = 1024
    cy.caustics_reflective = False
    cy.caustics_refractive = False
    cy.sample_clamp_direct = 0.0
    cy.sample_clamp_indirect = 8.0
    cy.blur_glossy = 0.5
    cy.pixel_filter_type = "BLACKMAN_HARRIS"
    cy.filter_width = 1.5
    if threads:
        scene.render.threads_mode = "FIXED"
        scene.render.threads = threads
    scene.render.use_persistent_data = True
    # motion blur on (house style); nothing moves in the still, so it is free
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = 0.5


def configure_output(scene, *, width, height, path, exposure=0.0, look="AgX - Punchy"):
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.filepath = path
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "16"
    scene.render.image_settings.compression = 60
    vs = scene.view_settings
    vs.view_transform = "AgX"
    try:
        vs.look = look
    except TypeError:
        vs.look = "None"
    vs.exposure = exposure
    vs.gamma = 1.0
    scene.display_settings.display_device = "sRGB"


def build_compositor(scene, *, bloom_strength=0.22, bloom_size=0.55, bloom_threshold=1.2, dispersion=0.012, distortion=-0.004, vignette=0.28):
    ng = bpy.data.node_groups.new("Compositing", "CompositorNodeTree")
    ng.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    scene.compositing_node_group = ng
    scene.use_nodes = True
    scene.render.use_compositing = True
    nodes, links = ng.nodes, ng.links

    rl = nodes.new("CompositorNodeRLayers")
    rl.scene = scene
    img = rl.outputs["Image"]

    # --- bloom ---------------------------------------------------------------
    glare = nodes.new("CompositorNodeGlare")
    try:
        glare.inputs["Type"].default_value = "Bloom"
    except Exception as e:  # noqa: BLE001
        print("[postfx] glare type:", e)
    for name, val in (("Quality", "High"),):
        try:
            glare.inputs[name].default_value = val
        except Exception:  # noqa: BLE001
            pass
    for name, val in (("Threshold", bloom_threshold), ("Strength", bloom_strength), ("Size", bloom_size), ("Saturation", 0.9), ("Smoothness", 0.15)):
        if name in glare.inputs:
            glare.inputs[name].default_value = val
    links.new(img, glare.inputs["Image"])
    img = glare.outputs["Image"]

    # a second, tighter/brighter halo pass for the hottest points (lamps, beam core)
    glare2 = nodes.new("CompositorNodeGlare")
    try:
        glare2.inputs["Type"].default_value = "Bloom"
        glare2.inputs["Quality"].default_value = "High"
    except Exception:  # noqa: BLE001
        pass
    for name, val in (("Threshold", 4.0), ("Strength", 0.12), ("Size", 0.25), ("Saturation", 0.8), ("Smoothness", 0.1)):
        if name in glare2.inputs:
            glare2.inputs[name].default_value = val
    links.new(img, glare2.inputs["Image"])
    img = glare2.outputs["Image"]

    # --- lens: faint chromatic aberration / barrel ------------------------------
    ld = nodes.new("CompositorNodeLensdist")
    try:
        ld.inputs["Type"].default_value = "Radial"
    except Exception:  # noqa: BLE001
        pass
    ld.inputs["Distortion"].default_value = distortion
    ld.inputs["Dispersion"].default_value = dispersion
    ld.inputs["Fit"].default_value = True
    links.new(img, ld.inputs["Image"])
    img = ld.outputs["Image"]

    # --- vignette: blurred ellipse mask multiplied in -----------------------------
    if vignette > 0:
        em = nodes.new("CompositorNodeEllipseMask")
        try:
            em.inputs["Size"].default_value = (0.92, 0.92)
            em.inputs["Position"].default_value = (0.5, 0.5)
        except Exception as e:  # noqa: BLE001
            print("[postfx] ellipse mask:", e)
        blur = nodes.new("CompositorNodeBlur")
        try:
            blur.inputs["Size"].default_value = (0.45 * scene.render.resolution_x, 0.45 * scene.render.resolution_x)
        except Exception as e:  # noqa: BLE001
            print("[postfx] blur:", e)
        links.new(em.outputs["Mask"], blur.inputs["Image"])
        # factor = 1 - vignette*(1-mask)
        inv = nodes.new("ShaderNodeMath")
        inv.operation = "SUBTRACT"
        inv.inputs[0].default_value = 1.0
        links.new(blur.outputs["Image"], inv.inputs[1])
        sc = nodes.new("ShaderNodeMath")
        sc.operation = "MULTIPLY_ADD"
        links.new(inv.outputs[0], sc.inputs[0])
        sc.inputs[1].default_value = -vignette
        sc.inputs[2].default_value = 1.0
        mul = nodes.new("ShaderNodeMix")
        mul.data_type = "RGBA"
        mul.blend_type = "MULTIPLY"
        C.sock_in(mul, "Factor_Float").default_value = 1.0
        links.new(img, C.sock_in(mul, "A_Color"))
        # broadcast factor to colour via a Combine node
        comb = nodes.new("ShaderNodeCombineColor")
        for ch in ("Red", "Green", "Blue"):
            links.new(sc.outputs[0], comb.inputs[ch])
        links.new(comb.outputs["Color"], C.sock_in(mul, "B_Color"))
        img = C.sock_out(mul, "Result_Color")

    out = nodes.new("NodeGroupOutput")
    links.new(img, out.inputs[0])
    return ng
