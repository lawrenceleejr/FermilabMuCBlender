# FermilabMuCBlender

Procedurally generated Blender scenes of the **Fermilab campus** — real
terrain and real OpenStreetMap building footprints, Wilson Hall highlighted —
with a **10 TeV muon collider** underneath, following the Fermilab
"site-filler" siting concept with a four-stage rapid-cycling-synchrotron
(RCS) acceleration chain.

Everything is built headlessly in GitHub Actions and uploaded as workflow
artifacts: two `.blend` files (a dark schematic style and a semi-realistic
golden-hour style with a quarter-disk cutaway into the geology) plus 1440p
Cycles renders from several named cameras, graded in the compositor (AgX
punchy look, mist-based aerial perspective, fog-glow on the machines, soft
vignette).

Each `.blend` also contains **`Cam_Tour`** — a 30 s keyframed camera
(720 frames @ 24 fps) that tours the complex: wide reveal, dive to Wilson
Hall, hero orbit, then a sweep over the excavated cutaway with the rings
below. It is not rendered in CI; render it locally with

```bash
blender -b fermilab_muc_realistic.blend -a          # PNG frames
# or -o //tour_ -F FFMPEG for a video, GPU strongly recommended
```

## The physics layout

Numbers follow the US Muon Collider white paper for the ESPPU
([arXiv:2503.23695](https://arxiv.org/abs/2503.23695)) and the IMCC interim
report ([arXiv:2407.12450](https://arxiv.org/abs/2407.12450)):

| Machine | Circumference | Energy | Notes |
|---|---|---|---|
| Linac + RLAs | — | → 63 GeV | near the existing proton complex |
| RCS1 | 6.28 km | 63 → 450 GeV | reuses the Tevatron tunnel footprint |
| RCS2 | 10.5 km | 0.45 → 1.725 TeV | new tunnel, ~100 m deep |
| RCS3 | 16.5 km | 1.725 → 3.56 TeV | shared largest tunnel, ~150 m deep |
| RCS4 | 16.5 km | 3.56 → 5.0 TeV | same tunnel as RCS3 |
| Collider ring | ~10 km | 10 TeV c.o.m. | ~200 m deep, 2 IP caverns |

The 16.5 km ring (diameter 5.25 km) is "almost exactly the size of the
campus": centered at the site's max-clearance point it still pokes ~110 m
past the fence at the tightest spot, faithful to the white paper. The deep
rings sit in the Galena–Platteville dolomite (bottom of layer ≈ 207 m below
grade), which the cutaway face hints at with a till/dolomite color split.

**Stated simplifications:** rings are drawn as perfect circles (the real
machines are racetracks with straight sections); tunnel tube radii are
visually exaggerated (12–18 m) so they read from aerial viewpoints; RCS4 is
drawn 45 m beside RCS3 to keep both visible in top view; the front-end
placement (target hall, cooling channel, linac) is schematic.

## Pipeline

```
scripts/fetch_data.py     one-time: terrain tiles + Overpass OSM -> data/   (committed snapshot)
scripts/build_scene.py    bpy: builds the scene, saves .blend    (runs in Blender, no network)
scripts/render_stills.py  bpy: renders every named camera        (Cycles CPU)
config/facility.py        every geometry knob in one place
```

- **Terrain**: AWS terrarium elevation tiles resampled to a 512×512 ENU
  grid (`data/heightmap.npz`); the realistic style extrudes it into a solid
  block and carves a quarter-disk pit (pure numpy masking — no booleans).
- **Buildings**: ~11k OSM footprints extruded to tagged/estimated heights,
  joined into one mesh; Wilson Hall stays separate with a highlight
  material (`data/buildings.geojson`).
- **Machines**: bezier-circle curves with bevel (exact circles, tiny
  memory), emissive Okabe–Ito color coding, plus shafts, IP caverns and
  detectors, target hall, and a legend.
- Data sources and licenses: see [`data/PROVENANCE.md`](data/PROVENANCE.md).

## Build locally

```bash
# Blender 4.5 LTS
blender --background --factory-startup --python scripts/build_scene.py -- \
    --style schematic --out out/fermilab_muc_schematic.blend        # or realistic
blender --background out/fermilab_muc_schematic.blend \
    --python scripts/render_stills.py -- --outdir out/renders
```

`--fast` on `build_scene.py` bakes low-res render settings (640×360,
32 samples) for quick iteration; `--no-cutaway` keeps the realistic terrain
intact. To refresh the geodata snapshot:
`pip install -r requirements-fetch.txt && python3 scripts/fetch_data.py`.

## CI

`.github/workflows/build.yml` runs on every push and via manual dispatch:
installs pinned Blender 4.5.11, builds both styles in a matrix, renders all
cameras with Cycles CPU, and uploads `fermilab-muc-schematic` /
`fermilab-muc-realistic` artifacts containing the `.blend` and PNGs.
