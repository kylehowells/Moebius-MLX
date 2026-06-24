#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="${MOEBIUS_MLX_MODEL_DIR:-/Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX}"
OUTPUT_DIR=""
INCLUDE_SOURCE_UNET=0
variants=()

usage() {
	cat <<'USAGE'
Usage: Scripts/stage_models_for_ios.sh --output PATH [--include-source-unet] [variant ...]

Stages converted model directories under:

  PATH/Moebius-MLX/<variant>

Copy the Moebius-MLX folder into the demo app's Documents container on a
physical iPad through Finder file sharing, or package the folder into the app
bundle for a custom build.

If no variants are supplied, ft_places2-f16 and ft_places2-q4 are staged.

Quantized variants are staged as runtime packages by default: manifest, VAE
files, and unet_quantized.safetensors. The original unet.safetensors is not
copied because the loader does not use it when unetQuantized is present.

Environment:
  MOEBIUS_MLX_MODEL_DIR  Override converted model root.
USAGE
}

while [ "$#" -gt 0 ]; do
	case "$1" in
		--models-root)
			shift
			if [ "$#" -eq 0 ]; then
				usage
				exit 2
			fi
			MODEL_ROOT="$1"
			;;
		--output|-o)
			shift
			if [ "$#" -eq 0 ]; then
				usage
				exit 2
			fi
			OUTPUT_DIR="$1"
			;;
		--include-source-unet)
			INCLUDE_SOURCE_UNET=1
			;;
		--help|-h)
			usage
			exit 0
			;;
		*)
			variants+=("$1")
			;;
	esac
	shift
done

if [ -z "$OUTPUT_DIR" ]; then
	usage
	exit 2
fi

if [ "${#variants[@]}" -eq 0 ]; then
	variants=(ft_places2-f16 ft_places2-q4)
fi

mkdir -p "$OUTPUT_DIR/Moebius-MLX"

manifest_value() {
	python3 - "$1" "$2" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1]))
value = manifest
for part in sys.argv[2].split("."):
	value = value.get(part)
	if value is None:
		break
if value is not None:
	print(value)
PY
}

copy_required_file() {
	local source_dir="$1"
	local destination_dir="$2"
	local filename="$3"
	if [ -z "$filename" ]; then
		return
	fi
	if [ ! -f "$source_dir/$filename" ]; then
		printf 'Missing required model file: %s\n' "$source_dir/$filename" >&2
		exit 1
	fi
	cp "$source_dir/$filename" "$destination_dir/$filename"
}

for variant in "${variants[@]}"; do
	source_dir="$MODEL_ROOT/$variant"
	destination_dir="$OUTPUT_DIR/Moebius-MLX/$variant"
	manifest="$source_dir/manifest.json"
	if [ ! -f "$manifest" ]; then
		printf 'Missing converted model variant: %s\n' "$source_dir" >&2
		exit 1
	fi
	rm -rf "$destination_dir"
	mkdir -p "$destination_dir"

	cp "$manifest" "$destination_dir/manifest.json"
	unet="$(manifest_value "$manifest" "files.unet")"
	unet_quantized="$(manifest_value "$manifest" "files.unetQuantized")"
	vae_encoder="$(manifest_value "$manifest" "files.vaeEncoder")"
	vae_decoder="$(manifest_value "$manifest" "files.vaeDecoder")"
	vae_encoder_quantized="$(manifest_value "$manifest" "files.vaeEncoderQuantized")"
	vae_decoder_quantized="$(manifest_value "$manifest" "files.vaeDecoderQuantized")"

	if [ -n "$unet_quantized" ]; then
		copy_required_file "$source_dir" "$destination_dir" "$unet_quantized"
		if [ "$INCLUDE_SOURCE_UNET" -eq 1 ]; then
			copy_required_file "$source_dir" "$destination_dir" "$unet"
		fi
	else
		copy_required_file "$source_dir" "$destination_dir" "$unet"
	fi
	if [ -n "$vae_encoder_quantized" ] && [ -n "$vae_decoder_quantized" ]; then
		copy_required_file "$source_dir" "$destination_dir" "$vae_encoder_quantized"
		copy_required_file "$source_dir" "$destination_dir" "$vae_decoder_quantized"
	else
		copy_required_file "$source_dir" "$destination_dir" "$vae_encoder"
		copy_required_file "$source_dir" "$destination_dir" "$vae_decoder"
	fi

	size="$(du -sh "$destination_dir" | awk '{ print $1 }')"
	printf 'staged %s (%s)\n' "$variant" "$size"
done

printf '\nStaged model folder:\n%s\n' "$OUTPUT_DIR/Moebius-MLX"
