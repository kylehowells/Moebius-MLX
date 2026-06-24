#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_FAMILY="ft_places2"
REFERENCE_MODEL="ft_places2-f16"
REFERENCE_LABEL="f16-image"
STEPS="2"
SAMPLES="gradient-circle"
IMAGE=""
MASK=""
PRECISIONS="f16,q4"
VARIANTS=""

usage() {
	cat <<USAGE
Usage: Scripts/benchmark_image.sh [options]

Options:
  --family NAME          Model family. Default: ft_places2.
  --reference-model ARG  Reference model name or directory. Default: ft_places2-f16.
  --reference-label NAME Reference label printed in rows. Default: f16-image.
  --steps N              Denoising steps. Default: 2.
  --samples LIST         Comma-separated deterministic image samples, or all.
                         Default: gradient-circle.
  --image PATH           Optional real image path. Requires --mask and ignores --samples.
  --mask PATH            Optional real mask path. White pixels are inpainted.
  --precisions LIST      Comma-separated precision list. Default: f16,q4.
  --variants LIST        Comma-separated variant entries. Overrides --precisions.
                         Entry formats:
                           label=model
                           label=model|unet_quantized_file|pointwise_policy|spatial_policy|vae_encoder_file|vae_decoder_file|vae_policy
USAGE
}

while [ "$#" -gt 0 ]; do
	case "$1" in
		--family)
			MODEL_FAMILY="$2"
			shift 2
			;;
		--reference-model)
			REFERENCE_MODEL="$2"
			shift 2
			;;
		--reference-label)
			REFERENCE_LABEL="$2"
			shift 2
			;;
		--steps)
			STEPS="$2"
			shift 2
			;;
		--samples)
			SAMPLES="$2"
			shift 2
			;;
		--image)
			IMAGE="$2"
			shift 2
			;;
		--mask)
			MASK="$2"
			shift 2
			;;
		--precisions)
			PRECISIONS="$2"
			shift 2
			;;
		--variants)
			VARIANTS="$2"
			shift 2
			;;
		--help|-h)
			usage
			exit 0
			;;
		*)
			printf 'Unknown option: %s\n\n' "$1" >&2
			usage >&2
			exit 2
			;;
	esac
done

(
	cd "$ROOT_DIR"
	swift build --product moebius-mlx >/dev/null
)
CLI="$ROOT_DIR/.build/debug/moebius-mlx"
MODEL_ROOT="/Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX"
TEMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/moebius-image-benchmark-variants.XXXXXX")"
trap 'rm -rf "$TEMP_ROOT"' EXIT

resolve_model_dir() {
	local model="$1"
	if [ -d "$model" ]; then
		printf '%s\n' "$model"
	elif [ -d "$MODEL_ROOT/$model" ]; then
		printf '%s\n' "$MODEL_ROOT/$model"
	else
		printf '%s\n' "$model"
	fi
}

variant_model_dir() {
	local label="$1"
	local model="$2"
	local unet_file="$3"
	local pointwise_policy="$4"
	local spatial_policy="$5"
	local vae_encoder_file="$6"
	local vae_decoder_file="$7"
	local vae_policy="$8"
	local source_dir
	source_dir="$(resolve_model_dir "$model")"
	if [ -z "$unet_file" ]; then
		printf '%s\n' "$source_dir"
		return
	fi
	if [ ! -f "$source_dir/manifest.json" ]; then
		printf 'Missing manifest for variant %s in %s\n' "$label" "$source_dir" >&2
		exit 1
	fi
	if [ ! -f "$source_dir/$unet_file" ]; then
		printf 'Missing UNet file for variant %s: %s\n' "$label" "$source_dir/$unet_file" >&2
		exit 1
	fi
	local view_dir="$TEMP_ROOT/$label"
	mkdir -p "$view_dir"
	python3 - "$source_dir" "$view_dir" "$unet_file" "$pointwise_policy" "$spatial_policy" "$vae_encoder_file" "$vae_decoder_file" "$vae_policy" <<'PY'
import json
import os
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
unet_file = sys.argv[3]
pointwise_policy = sys.argv[4]
spatial_policy = sys.argv[5]
vae_encoder_file = sys.argv[6]
vae_decoder_file = sys.argv[7]
vae_policy = sys.argv[8]

for child in source.iterdir():
    target_child = target / child.name
    if child.name == "manifest.json" or target_child.exists():
        continue
    os.symlink(child, target_child)

manifest = json.loads((source / "manifest.json").read_text())
manifest.setdefault("files", {})["unetQuantized"] = unet_file
if pointwise_policy and pointwise_policy != "none":
    manifest["quantizedPointwiseConvolutions"] = True
    manifest["quantizedPointwiseConvolutionPolicy"] = pointwise_policy
else:
    manifest.pop("quantizedPointwiseConvolutions", None)
    manifest.pop("quantizedPointwiseConvolutionPolicy", None)
if spatial_policy and spatial_policy != "none":
    manifest["quantizedSpatialConvolutions"] = True
    manifest["quantizedSpatialConvolutionPolicy"] = spatial_policy
else:
    manifest.pop("quantizedSpatialConvolutions", None)
    manifest.pop("quantizedSpatialConvolutionPolicy", None)
if vae_encoder_file and vae_decoder_file and vae_policy and vae_policy != "none":
    manifest["files"]["vaeEncoderQuantized"] = vae_encoder_file
    manifest["files"]["vaeDecoderQuantized"] = vae_decoder_file
    manifest["quantizedVAEConvolutions"] = True
    manifest["quantizedVAEConvolutionPolicy"] = vae_policy
else:
    manifest["files"].pop("vaeEncoderQuantized", None)
    manifest["files"].pop("vaeDecoderQuantized", None)
    manifest.pop("quantizedVAEConvolutions", None)
    manifest.pop("quantizedVAEConvolutionPolicy", None)
(target / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY
	printf '%s\n' "$view_dir"
}

printf 'variant\tmodel_mb\tmax_abs\tmean_abs\trmse\tp99_abs\tpsnr_db\tcosine\tinference_seconds\tload_seconds\ttotal_seconds\n'
if [ -n "$VARIANTS" ]; then
	IFS=',' read -r -a VARIANT_ARRAY <<< "$VARIANTS"
else
	IFS=',' read -r -a PRECISION_ARRAY <<< "$PRECISIONS"
	VARIANT_ARRAY=()
	for precision in "${PRECISION_ARRAY[@]}"; do
		VARIANT_ARRAY+=("${MODEL_FAMILY}-${precision}=${MODEL_FAMILY}-${precision}")
	done
fi

reference_argument="$(resolve_model_dir "$REFERENCE_MODEL")"
for entry in "${VARIANT_ARRAY[@]}"; do
	if [[ "$entry" != *=* ]]; then
		printf 'Invalid variant entry: %s\n' "$entry" >&2
		exit 2
	fi
	variant="${entry%%=*}"
	spec="${entry#*=}"
	IFS='|' read -r model_spec unet_file pointwise_policy spatial_policy vae_encoder_file vae_decoder_file vae_policy _ <<< "$spec"
	model_argument="$(variant_model_dir "$variant" "$model_spec" "${unet_file:-}" "${pointwise_policy:-}" "${spatial_policy:-}" "${vae_encoder_file:-}" "${vae_decoder_file:-}" "${vae_policy:-}")"
	output="$(
		image_args=()
		if [ -n "$IMAGE" ] || [ -n "$MASK" ]; then
			image_args=(--image "$IMAGE" --mask "$MASK")
		fi
		"$CLI" benchmark-image "$model_argument" \
			--reference-model "$reference_argument" \
			--reference-name "$REFERENCE_LABEL" \
			--steps "$STEPS" \
			--samples "$SAMPLES" \
			"${image_args[@]}"
	)"
	model_mb="$(printf '%s\n' "$output" | awk '/^model_mb_total:/ { print $2 }')"
	max_abs="$(printf '%s\n' "$output" | awk '/^max_abs:/ { print $2 }')"
	mean_abs="$(printf '%s\n' "$output" | awk '/^mean_abs:/ { print $2 }')"
	rmse="$(printf '%s\n' "$output" | awk '/^rmse:/ { print $2 }')"
	p99_abs="$(printf '%s\n' "$output" | awk '/^p99_abs:/ { print $2 }')"
	psnr_db="$(printf '%s\n' "$output" | awk '/^psnr_db:/ { print $2 }')"
	cosine="$(printf '%s\n' "$output" | awk '/^cosine_similarity:/ { print $2 }')"
	inference_seconds="$(printf '%s\n' "$output" | awk '/^inference_seconds:/ { print $2 }')"
	load_seconds="$(printf '%s\n' "$output" | awk '/^load_seconds:/ { print $2 }')"
	total_seconds="$(printf '%s\n' "$output" | awk '/^total_seconds:/ { print $2 }')"
	printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
		"$variant" "$model_mb" "$max_abs" "$mean_abs" "$rmse" "$p99_abs" "$psnr_db" "$cosine" "$inference_seconds" "$load_seconds" "$total_seconds"
done
