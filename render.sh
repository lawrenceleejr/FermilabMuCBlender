#!/usr/bin/env bash
# Render the Fermilab muon-collider hero image with Blender (>= 5) in Cycles.
#
#   ./render.sh                                   # final 2560x1440 hero
#   ./render.sh --preview                         # fast 640x360 look-dev pass
#   ./render.sh --camera portrait --res 1600x2000 # cover-format variant
#   BLENDER=/opt/blender/blender ./render.sh --samples 1024
#
# Any other flags are forwarded to scene/build_scene.py (see its docstring).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BLENDER="${BLENDER:-blender}"

command -v "$BLENDER" >/dev/null 2>&1 || { echo "blender not found; set BLENDER=/path/to/blender" >&2; exit 1; }
[[ -d "$HERE/assets/textures/Grass004_2K-JPG" ]] || python3 "$HERE/tools/fetch_assets.py"

args=()
for a in "$@"; do
  case "$a" in
    --preview) args+=(--res 640x360 --samples 48 --tree-density 0.35 --out "$HERE/out/preview.png") ;;
    *) args+=("$a") ;;
  esac
done
exec "$BLENDER" -b -P "$HERE/scene/build_scene.py" -- "${args[@]}"
