#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FAMILY="ft_places2"
REFERENCE_MODEL="ft_places2-f16"
REFERENCE_LABEL="f16-image"
STEPS="2"
SAMPLES="gradient-circle"
IMAGE=""
MASK=""
PRECISIONS="f16,q4"
VARIANTS=""
OUTPUT_DIR=""

usage() {
	cat <<USAGE
Usage: Scripts/run_image_benchmark_report.sh [options]

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
  --output-dir PATH      Report output directory.
USAGE
}

while [ "$#" -gt 0 ]; do
	case "$1" in
		--family)
			FAMILY="$2"
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
		--output-dir)
			OUTPUT_DIR="$2"
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

if [ -z "$OUTPUT_DIR" ]; then
	OUTPUT_DIR="/Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX-benchmarks/reports/${FAMILY}-image-${REFERENCE_LABEL}-steps${STEPS}-$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$OUTPUT_DIR"

args=(--family "$FAMILY" --reference-model "$REFERENCE_MODEL" --reference-label "$REFERENCE_LABEL" --steps "$STEPS" --samples "$SAMPLES" --precisions "$PRECISIONS")
if [ -n "$VARIANTS" ]; then
	args+=(--variants "$VARIANTS")
fi
if [ -n "$IMAGE" ] || [ -n "$MASK" ]; then
	args+=(--image "$IMAGE" --mask "$MASK")
fi

"$ROOT_DIR/Scripts/benchmark_image.sh" "${args[@]}" > "$OUTPUT_DIR/benchmark.tsv"
title_suffix="samples=${SAMPLES}"
if [ -n "$IMAGE" ]; then
	title_suffix="image=$(basename "$IMAGE")"
fi
python3 "$ROOT_DIR/Scripts/plot_benchmark_report.py" \
	--input "$OUTPUT_DIR/benchmark.tsv" \
	--output-dir "$OUTPUT_DIR" \
	--title "Moebius MLX ${FAMILY} image ${REFERENCE_LABEL} steps=${STEPS} ${title_suffix}" \
	--fidelity-title "Image Fidelity Error Metrics" \
	--fidelity-ylabel "absolute RGB error, log scale"

printf 'report: %s\n' "$OUTPUT_DIR"
