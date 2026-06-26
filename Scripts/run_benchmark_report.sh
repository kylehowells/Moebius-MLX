#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_FAMILY="ft_places2"
REFERENCE="onnx"
FIXTURE_DIR=""
STEPS="20"
PRECISIONS="f32,f16,q8,q4,mxfp8,mxfp4"
VARIANTS=""
OUTPUT_DIR=""

usage() {
	cat <<USAGE
Usage: Scripts/run_benchmark_report.sh [options]

Options:
  --family NAME          Model family. Default: ft_places2.
  --reference NAME       Reference generator/label: onnx or pytorch. Default: onnx.
  --fixture PATH         Existing fixture directory. Skips reference generation.
  --steps N              Denoising steps for generated fixture. Default: 20.
  --precisions LIST      Comma-separated precision list. Default: f32,f16,q8,q4,mxfp8,mxfp4.
  --variants LIST        Comma-separated named variant entries. Overrides --precisions.
                         See Scripts/benchmark_fidelity.sh --help for formats.
  --output-dir PATH      Report output directory. Default: BenchmarkReports/<timestamp>.

Outputs:
  benchmark.tsv
  README.md
  fidelity_errors.png
  performance_timings.png
  model_size.png
  speed_fidelity_tradeoff.png
  size_fidelity_tradeoff.png
USAGE
}

while [ "$#" -gt 0 ]; do
	case "$1" in
		--family)
			MODEL_FAMILY="$2"
			shift 2
			;;
		--reference)
			REFERENCE="$2"
			shift 2
			;;
		--fixture)
			FIXTURE_DIR="$2"
			shift 2
			;;
		--steps)
			STEPS="$2"
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
	timestamp="$(date +%Y%m%d-%H%M%S)"
	OUTPUT_DIR="$ROOT_DIR/BenchmarkReports/${MODEL_FAMILY}-${REFERENCE}-steps${STEPS}-${timestamp}"
fi

mkdir -p "$OUTPUT_DIR"
TSV="$OUTPUT_DIR/benchmark.tsv"

benchmark_args=(
	--family "$MODEL_FAMILY"
	--reference "$REFERENCE"
	--steps "$STEPS"
	--precisions "$PRECISIONS"
)
if [ -n "$VARIANTS" ]; then
	benchmark_args+=(--variants "$VARIANTS")
fi
if [ -n "$FIXTURE_DIR" ]; then
	benchmark_args+=(--fixture "$FIXTURE_DIR")
fi

"$ROOT_DIR/Scripts/benchmark_fidelity.sh" "${benchmark_args[@]}" > "$TSV"

python3 "$ROOT_DIR/Scripts/plot_benchmark_report.py" \
	--input "$TSV" \
	--output-dir "$OUTPUT_DIR" \
	--title "Moebius MLX ${MODEL_FAMILY} ${REFERENCE} steps=${STEPS}" \
	--fidelity-ylabel "absolute latent error, log scale"

printf 'report: %s\n' "$OUTPUT_DIR"
