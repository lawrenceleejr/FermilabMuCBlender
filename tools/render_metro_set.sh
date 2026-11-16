#!/usr/bin/env bash
# Render the metro view's HQ set on a GPU workstation: the plain view and the
# four comparison rings, at the final settings (5760x2400, 1024 spp ceiling,
# 1 % adaptive noise threshold).
#
#   tools/render_metro_set.sh                 # all five
#   tools/render_metro_set.sh fcc ssc         # just these
#   BLENDER=/Applications/Blender.app/Contents/MacOS/Blender tools/render_metro_set.sh
#
# Placement: the FCC sits in the gap between the site and the city
# (--compare-at gap); the others are centred halfway between the collider and
# Willis Tower, as in the test renders. Override with COMPARE_AT=gap|middle.
#
# Each finished frame is written straight to renders/metro_<name>_hq.png; a
# frame already there is skipped, so an interrupted run resumes where it stopped.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -z "${BLENDER:-}" ]]; then
  if command -v blender >/dev/null 2>&1; then BLENDER=blender
  elif [[ -x /Applications/Blender.app/Contents/MacOS/Blender ]]; then BLENDER=/Applications/Blender.app/Contents/MacOS/Blender
  else echo "Blender not found: set BLENDER=/path/to/blender" >&2; exit 1; fi
fi
export BLENDER

sets=("$@")
[[ ${#sets[@]} -eq 0 ]] && sets=(plain lhc fcc ssc vlhc)

for name in "${sets[@]}"; do
  out="$HERE/renders/metro_${name}_hq.png"
  if [[ -s "$out" ]]; then echo "[metro-set] $out exists, skipping"; continue; fi
  extra=()
  if [[ "$name" != "plain" ]]; then
    at="${COMPARE_AT:-}"
    [[ -z "$at" ]] && { [[ "$name" == "fcc" ]] && at=gap || at=middle; }
    extra=(--compare "$name" --compare-at "$at")
  fi
  echo "[metro-set] $name -> $out"
  t0=$(date +%s)
  "$HERE/render.sh" --gpu --final --camera metro ${extra[@]+"${extra[@]}"} --out "$out"
  echo "[metro-set] $name done in $(( ($(date +%s) - t0) / 60 )) min"
done
