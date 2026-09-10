#!/usr/bin/env bash
# Render the Fermilab muon-collider hero image with Blender (>= 5) in Cycles.
#
#   ./render.sh --preview                         # fast 640x360 look-dev pass (CPU ok)
#   ./render.sh --gpu --final                     # 1024 spp; resolution follows the camera's aspect
#   ./render.sh --gpu --final --camera overview   # 3840x2560 (the overview frames 3:2, not 16:9)
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

# The overview camera frames a 3:2 plate, not 16:9: at 16:9 the same camera
# gets half the sky for the same ground, and the framing was solved against 3:2.
# So --preview and --final pick the aspect from the camera rather than assuming
# one, and an explicit --res always wins.
camera="northeast"; has_res=0; prev=""
for a in "$@"; do
  case "$prev" in --camera) camera="$a" ;; esac
  case "$a" in --res|--res=*) has_res=1 ;; esac
  prev="$a"
done
case "$camera" in
  overview|overview_south) final_res=3840x2560; prev_res=640x427 ;;
  cover|portrait)          final_res=2400x3600; prev_res=400x600 ;;
  *)                       final_res=3840x2160; prev_res=640x360 ;;
esac

args=()
for a in "$@"; do
  case "$a" in
    --preview)
      args+=(--samples 48 --tree-density 0.35 --out "$HERE/out/preview.png")
      if [[ $has_res -eq 0 ]]; then args+=(--res "$prev_res"); fi ;;
    --gpu)     args+=(--device GPU) ;;
    --final)
      args+=(--samples 1024 --adaptive-threshold 0.005)
      if [[ $has_res -eq 0 ]]; then args+=(--res "$final_res"); fi ;;
    *) args+=("$a") ;;
  esac
done
exec "$BLENDER" -b -P "$HERE/scene/build_scene.py" -- "${args[@]}"
