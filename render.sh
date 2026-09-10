#!/usr/bin/env bash
# Render the Fermilab muon-collider imagery with Blender (>= 5) in Cycles.
#
#   ./render.sh --preview                      # fast look-dev pass (CPU is fine)
#   ./render.sh --gpu --final                  # one still; resolution follows the camera
#   ./render.sh --gpu --both                   # the two deliverables: plain + annotated
#   ./render.sh --gpu --movie                  # 15 s camera-approach movie, no labels
#   ./render.sh --gpu --both --camera overview_south --out renders/cover.png
#   BLENDER=/opt/blender/blender ./render.sh --samples 1024
#
# First time on a machine:  python3 tools/fetch_assets.py
#                           python3 tools/fetch_geodata.py --essential
#                           python3 tools/fetch_dem.py && python3 tools/bake_geodata.py
#                           python3 tools/get_fonts.py
#                           blender -b --python tools/make_sky_hdri.py -- --res 8k
#
# Any other flags are forwarded to scene/build_scene.py (see its docstring).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BLENDER="${BLENDER:-blender}"

command -v "$BLENDER" >/dev/null 2>&1 || { echo "blender not found; set BLENDER=/path/to/blender" >&2; exit 1; }
[[ -d "$HERE/assets/textures/Grass004_2K-JPG" ]] || python3 "$HERE/tools/fetch_assets.py"
[[ -f "$HERE/assets/hdri/fermilab_night_sky.exr" ]] || "$BLENDER" -b --python "$HERE/tools/make_sky_hdri.py" -- --res 8k

# The overview camera frames a 3:2 plate, not 16:9: at 16:9 the same camera gets
# half the sky for the same ground, and the framing was solved against 3:2. So
# resolution follows the camera rather than being assumed, and an explicit --res
# always wins.
camera="northeast"; has_res=0; has_out=0; want_both=0; want_movie=0; prev=""
for a in "$@"; do
  case "$prev" in --camera) camera="$a" ;; esac
  case "$a" in
    --res|--res=*) has_res=1 ;;
    --out|--out=*) has_out=1 ;;
    --both)  want_both=1 ;;
    --movie) want_movie=1 ;;
  esac
  prev="$a"
done
case "$camera" in
  overview|overview_south) final_res=3840x2560; prev_res=640x427; movie_res=1920x1280 ;;
  cover|portrait)          final_res=2400x3600; prev_res=400x600;  movie_res=1200x1800 ;;
  *)                       final_res=3840x2160; prev_res=640x360;  movie_res=1920x1080 ;;
esac
default_out="$HERE/renders/fermilab_site_${camera}.png"

args=(); res_set=$has_res
for a in "$@"; do
  case "$a" in
    --preview)
      args+=(--samples 48 --tree-density 0.35 --out "$HERE/out/preview.png")
      if [[ $res_set -eq 0 ]]; then args+=(--res "$prev_res"); res_set=1; fi ;;
    --gpu)   args+=(--device GPU) ;;
    --final)
      args+=(--samples 1024 --adaptive-threshold 0.005)
      if [[ $res_set -eq 0 ]]; then args+=(--res "$final_res"); res_set=1; fi ;;
    --both|--movie) ;;                    # handled below, not forwarded
    *) args+=("$a") ;;
  esac
done

# --------------------------------------------------------------------- movie --
if [[ $want_movie -eq 1 ]]; then
  # No labels at all, by construction: --annotations is never passed, so no
  # anchors are dumped and nothing downstream can draw a callout.
  stem="$HERE/renders/movie_${camera}"
  margs=(--animate "${MOVIE_SECONDS:-15}" --fps "${MOVIE_FPS:-30}" --out "${stem}.png")
  if [[ $res_set -eq 0 ]]; then margs+=(--res "$movie_res"); fi
  # 450 frames at 1024 spp is days of GPU; 320 with a loose adaptive threshold
  # denoises to something a projector cannot tell apart on a moving image.
  margs+=(--samples "${MOVIE_SAMPLES:-320}" --adaptive-threshold 0.01)
  echo "[render.sh] movie: ${MOVIE_SECONDS:-15}s at ${MOVIE_FPS:-30} fps, camera $camera"
  "$BLENDER" -b -P "$HERE/scene/build_scene.py" -- "${args[@]}" "${margs[@]}"
  if command -v ffmpeg >/dev/null 2>&1; then
    echo "[render.sh] encoding ${stem}.mp4"
    ffmpeg -y -loglevel warning -framerate "${MOVIE_FPS:-30}" \
      -i "${stem}/frame_%04d.png" -c:v libx264 -pix_fmt yuv420p -crf 17 \
      -movflags +faststart "${stem}.mp4"
    echo "[render.sh] wrote ${stem}.mp4"
  else
    echo "[render.sh] ffmpeg not found; frames are in ${stem}/ -- encode with:"
    echo "  ffmpeg -y -framerate ${MOVIE_FPS:-30} -i ${stem}/frame_%04d.png \\"
    echo "    -c:v libx264 -pix_fmt yuv420p -crf 17 ${stem}.mp4"
  fi
  exit 0
fi

# ------------------------------------------------- both stills, plain first --
if [[ $want_both -eq 1 ]]; then
  out="$default_out"
  if [[ $has_out -eq 1 ]]; then
    prev=""
    for a in "$@"; do
      case "$prev" in --out) out="$a" ;; esac
      prev="$a"
    done
  fi
  anno="$HERE/out/$(basename "${out%.*}")_anno.json"
  bargs=(--out "$out" --annotations "$anno")
  if [[ $res_set -eq 0 ]]; then bargs+=(--res "$final_res"); fi
  echo "[render.sh] beauty pass -> $out"
  "$BLENDER" -b -P "$HERE/scene/build_scene.py" -- "${args[@]}" "${bargs[@]}"
  echo "[render.sh] annotation pass -> ${out%.*}_annotated.{png,pdf,svg}"
  python3 "$HERE/tools/annotate.py" "$out" --anno "$anno" --layout "$camera" \
    --out "${out%.*}_annotated" --formats png,pdf,svg --strict
  echo "[render.sh] both deliverables written:"
  echo "  unlabelled : $out"
  echo "  annotated  : ${out%.*}_annotated.png (+ .pdf, .svg)"
  exit 0
fi

exec "$BLENDER" -b -P "$HERE/scene/build_scene.py" -- "${args[@]}"
