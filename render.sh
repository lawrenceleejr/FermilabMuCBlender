#!/usr/bin/env bash
# Render the Fermilab muon-collider hero image with Blender (>= 5) in Cycles.
#
#   ./render.sh --preview                         # fast 640x360 look-dev pass (CPU ok)
#   ./render.sh --gpu --final                     # 3840x2160, 1024 spp on the workstation GPU
#   ./render.sh --gpu --final --camera portrait --res 2400x3000 --out renders/cover_portrait.png
#   BLENDER=/opt/blender/blender ./render.sh --samples 1024
#
# First time on a machine:  python3 tools/fetch_assets.py
#                           blender -b --python tools/make_sky_hdri.py -- --res 8k
#
# Any other flags are forwarded to scene/build_scene.py (see its docstring).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BLENDER="${BLENDER:-blender}"

command -v "$BLENDER" >/dev/null 2>&1 || { echo "blender not found; set BLENDER=/path/to/blender" >&2; exit 1; }
[[ -d "$HERE/assets/textures/Grass004_2K-JPG" ]] || python3 "$HERE/tools/fetch_assets.py"
[[ -f "$HERE/assets/hdri/fermilab_night_sky.exr" ]] || "$BLENDER" -b --python "$HERE/tools/make_sky_hdri.py" -- --res 8k

args=()
for a in "$@"; do
  case "$a" in
    --preview) args+=(--res 640x360 --samples 48 --tree-density 0.35 --out "$HERE/out/preview.png") ;;
    --gpu)     args+=(--device GPU) ;;
    --final)   args+=(--res 3840x2160 --samples 1024) ;;
    *) args+=("$a") ;;
  esac
done
exec "$BLENDER" -b -P "$HERE/scene/build_scene.py" -- "${args[@]}"
