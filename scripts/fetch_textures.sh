#!/usr/bin/env bash
#
# Download the CC0 terrain material textures (albedo + OpenGL normal) used by the
# palette "vibes" from ambientCG and normalize them into res/textures/ as
# 1024x1024 PNGs. Each palette pairs a flat (ground) material with a slope (rock)
# material; see src/main/materials.c for how these map to palettes.
#
# CC0 1.0 (public domain) — no attribution required; see res/textures/SOURCES.txt.
# Requires: curl, unzip, ffmpeg.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/res/textures"
SIZE=1024
mkdir -p "$OUT"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fetch() {
    local asset="$1" base="$2"
    local zip="$TMP/$asset.zip" ex="$TMP/$asset"
    echo "Downloading $asset -> $base ..."
    curl -fsSL "https://ambientcg.com/get?file=${asset}_1K-JPG.zip" -o "$zip"
    mkdir -p "$ex"
    unzip -o -q "$zip" -d "$ex"

    local color normal
    color="$(find "$ex" -iname '*color*' | head -1)"
    normal="$(find "$ex" -iname '*normalgl*' | head -1)"
    [ -n "$color" ]  || { echo "ERROR: no color map in $asset"  >&2; exit 1; }
    [ -n "$normal" ] || { echo "ERROR: no NormalGL map in $asset" >&2; exit 1; }

    ffmpeg -y -loglevel error -i "$color"  -vf "scale=${SIZE}:${SIZE}" "$OUT/${base}_albedo.png"
    ffmpeg -y -loglevel error -i "$normal" -vf "scale=${SIZE}:${SIZE}" "$OUT/${base}_normal.png"
    echo "  -> ${base}_albedo.png, ${base}_normal.png  (from $asset)"
}

# Start clean so renamed/removed materials don't leave orphans behind.
rm -f "$OUT"/*_albedo.png "$OUT"/*_normal.png

# material name        ambientCG asset (CC0)
fetch Grass004 grass          # green grass     (Meadow flat)
fetch Rock023  slate_gray     # gray slate      (Meadow slope)
fetch Ground068 dirt_badlands # brown-green dirt(Badlands flat)
fetch Rock029  slate_red      # reddish rock    (Badlands slope)
fetch Ground054 dirt_tan      # tan dirt        (Tundra flat)
fetch Rock031  slate_dark     # dark charcoal   (Tundra slope)
fetch Ground078 dirt_pebble   # red-brown pebble(Mesa flat)
fetch Rock017  slate_white    # white/cream rock(Mesa slope)

cat > "$OUT/SOURCES.txt" <<EOF
Terrain material textures
=========================
Source: ambientCG (https://ambientcg.com) — License: CC0 1.0 (public domain).
No attribution required; recorded here for provenance.

Palette "vibes" (flat ground + slope rock), see src/main/materials.c:
  Meadow   : grass         <- Grass004 | slate_gray  <- Rock023
  Badlands : dirt_badlands <- Ground068| slate_red   <- Rock029
  Tundra   : dirt_tan      <- Ground054| slate_dark  <- Rock031
  Mesa     : dirt_pebble   <- Ground078| slate_white <- Rock017

All maps are Color + NormalGL at 1K. Regenerate with: scripts/fetch_textures.sh
EOF

echo "Done. Wrote textures to $OUT"
