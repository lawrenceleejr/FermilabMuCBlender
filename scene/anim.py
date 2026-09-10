"""Scene animation: lowering the proposed machines onto the site.

The camera move lives in `camera_rig.animate_approach`; this is the other half
of the movie. The proposed rings descend onto the ground while the camera
rises, so the sequence reads as the machine being placed on the site rather
than as a flythrough of something already there.

Only the *proposed* machines move. The Tevatron and the Main Injector are
existing plant and are already in the ground, which is the same distinction the
still figure makes with colour -- letting them drop in too would say something
false about the site.

Each group gets its own start height, start frame and duration, so they do not
arrive as a set: at the midpoint of the clip the six are at 212, 105, 385, 282,
411 and 350 m, and they touch down across four seconds rather than together.
That stagger is what makes the move feel organic; it is not a physical
simulation, and the numbers do not pretend to be one -- the outer rings start
highest and take longest but travel faster in m/s, which is the opposite of
what mass would do. Fifteen seconds is not enough clip to have both a tall drop
and a slow one.
"""
from __future__ import annotations

import bpy

from . import common as C

# (label, name prefixes, start height m, start fraction, land fraction)
# Fractions are of the whole clip. Everything lands by 0.92 so the last tenth
# of the movie is settled and its final frame matches the still exactly.
DESCENT = (
    ("RCS 4 filler", ("rcs4_tunnel", "rcs4_rf_"), 1500.0, 0.00, 0.66),
    ("collider", ("muon_collider_ring", "muon_collider_sheath"), 1150.0, 0.08, 0.60),
    ("RCS 3", ("rcs3_tunnel", "rcs3_rf_"), 900.0, 0.16, 0.78),
    ("RCS 1-2", ("rcs12_tunnel", "rcs12_rf_"), 700.0, 0.24, 0.70),
    ("detector halls", ("detector_hall_",), 520.0, 0.34, 0.86),
    ("front end", ("cooling_", "pd_", "proton_driver_linac", "target_hall"),
     380.0, 0.40, 0.92),
)


def animate_descent(scene, frames, *, groups=DESCENT):
    """Keyframe the proposed machines descending onto the ground.

    Returns a list of (label, object count, start height, start frame, land
    frame) so the caller can report what actually moved -- the groups are
    matched by object-name prefix, and a builder rename would otherwise leave a
    group silently empty and the ring simply sitting there.
    """
    report = []
    for label, prefixes, z0, f0, f1 in groups:
        objs = [o for o in bpy.data.objects if o.name.startswith(prefixes)]
        if not objs:
            print(f"[anim] WARNING: no objects match {prefixes} for {label!r}; "
                  "nothing will descend for that group (builder renamed?)")
            continue
        start = max(int(round(f0 * frames)), 1)
        land = min(max(int(round(f1 * frames)), start + 2), frames)
        for o in objs:
            base = o.location.copy()
            o.location = (base.x, base.y, base.z + z0)
            o.keyframe_insert(data_path="location", frame=1)
            o.keyframe_insert(data_path="location", frame=start)
            o.location = base
            o.keyframe_insert(data_path="location", frame=land)
            for fc in C.action_fcurves(o.animation_data.action):
                for kp in fc.keyframe_points:
                    kp.interpolation = "SINE"
                    kp.easing = "EASE_IN_OUT"
                fc.update()
        report.append((label, len(objs), z0, start, land))
    return report
