#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/build_mlx_metallib.sh [debug|release]

Builds MLX's Metal shader library (mlx.metallib) and copies it into the SwiftPM output directory.

If you see "missing Metal Toolchain", run:
  xcodebuild -downloadComponent MetalToolchain
USAGE
}

CONFIG="${1:-debug}"
if [[ "$CONFIG" != "release" && "$CONFIG" != "debug" ]]; then
  usage
  exit 2
fi
case "$CONFIG" in
  debug)
    CONFIG_CAP="Debug"
    ;;
  release)
    CONFIG_CAP="Release"
    ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/.build"
MLX_SWIFT_DIR="$BUILD_DIR/checkouts/mlx-swift"
KERNELS_DIR="$MLX_SWIFT_DIR/Source/Cmlx/mlx/mlx/backend/metal/kernels"

if [[ ! -d "$BUILD_DIR" ]]; then
  echo "error: $BUILD_DIR not found (run swift build first)" >&2
  exit 1
fi

if [[ ! -d "$KERNELS_DIR" ]]; then
  echo "error: MLX metal kernels not found at $KERNELS_DIR" >&2
  echo "hint: run swift build once to fetch dependencies" >&2
  exit 1
fi

OUT_DIRS=()
while IFS= read -r DIR; do
  OUT_DIRS+=("$DIR")
done < <(find "$BUILD_DIR" -maxdepth 3 -type d \( -path "*/$CONFIG" -o -path "*/$CONFIG_CAP" \) | LC_ALL=C sort)
if [[ "${#OUT_DIRS[@]}" -eq 0 ]]; then
  echo "error: could not find SwiftPM output directory for config=$CONFIG" >&2
  exit 1
fi

METAL_SRCS=()
while IFS= read -r SRC; do
  METAL_SRCS+=("$SRC")
done < <(find "$KERNELS_DIR" -type f -name '*.metal' ! -name '*_nax.metal' | LC_ALL=C sort)
if [[ "${#METAL_SRCS[@]}" -eq 0 ]]; then
  echo "error: no .metal files found under $KERNELS_DIR" >&2
  exit 1
fi

TMP="$(mktemp -d "${TMPDIR:-/tmp}/mlx-metallib.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

METAL_FLAGS=(
  -x metal
  -Wall
  -Wextra
  -fno-fast-math
  -Wno-c++17-extensions
  -Wno-c++20-extensions
)

echo "Compiling ${#METAL_SRCS[@]} Metal sources..."
AIR_FILES=()
INDEX=0
for SRC in "${METAL_SRCS[@]}"; do
  INDEX=$((INDEX + 1))
  REL="${SRC#"$KERNELS_DIR/"}"
  KEY="$(printf '%s' "$REL" | shasum -a 256 | awk '{print $1}' | cut -c1-16)"
  OUT_AIR="$TMP/$KEY.air"
  echo "[$INDEX/${#METAL_SRCS[@]}] metal $REL"

  if ! xcrun -sdk macosx metal "${METAL_FLAGS[@]}" -c "$SRC" -I"$KERNELS_DIR" -I"$MLX_SWIFT_DIR/Source/Cmlx/mlx" -o "$OUT_AIR" 2>"$TMP/metal.err"; then
    if grep -q "missing Metal Toolchain" "$TMP/metal.err" 2>/dev/null; then
      echo "error: missing Metal Toolchain" >&2
      echo "run: xcodebuild -downloadComponent MetalToolchain" >&2
    fi
    cat "$TMP/metal.err" >&2
    exit 1
  fi

  AIR_FILES+=("$OUT_AIR")
done

OUT_METALLIB="$TMP/default.metallib"
xcrun -sdk macosx metallib "${AIR_FILES[@]}" -o "$OUT_METALLIB"
for OUT_DIR in "${OUT_DIRS[@]}"; do
  cp "$OUT_METALLIB" "$OUT_DIR/default.metallib"
  cp "$OUT_METALLIB" "$OUT_DIR/mlx.metallib"
  echo "OK: wrote $OUT_DIR/default.metallib"
  while IFS= read -r BUNDLE_DIR; do
    MACOS_DIR="$BUNDLE_DIR/Contents/MacOS"
    RESOURCES_DIR="$BUNDLE_DIR/Contents/Resources"
    if [[ -d "$MACOS_DIR" ]]; then
      cp "$OUT_METALLIB" "$MACOS_DIR/mlx.metallib"
      cp "$OUT_METALLIB" "$MACOS_DIR/default.metallib"
      echo "OK: wrote $MACOS_DIR/mlx.metallib"
    fi
    mkdir -p "$RESOURCES_DIR"
    cp "$OUT_METALLIB" "$RESOURCES_DIR/mlx.metallib"
    cp "$OUT_METALLIB" "$RESOURCES_DIR/default.metallib"
    echo "OK: wrote $RESOURCES_DIR/default.metallib"
  done < <(find "$OUT_DIR" -maxdepth 1 -type d -name '*.xctest' | LC_ALL=C sort)
done
