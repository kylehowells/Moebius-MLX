#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${MOEBIUS_MLX_MODEL_DIR:-/Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX}"
RUN_DEMO_BUILDS=0
RUN_ALL_INSPECTS=0
FIDELITY_FIXTURE=""
FIDELITY_REFERENCE="onnx"

usage() {
	printf 'Usage: %s [--models-root PATH] [--demo-builds] [--all-inspects] [--fidelity-fixture PATH] [--fidelity-reference NAME]\n' "$0"
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
		--demo-builds)
			RUN_DEMO_BUILDS=1
			;;
		--all-inspects)
			RUN_ALL_INSPECTS=1
			;;
		--fidelity-fixture)
			shift
			if [ "$#" -eq 0 ]; then
				usage
				exit 2
			fi
			FIDELITY_FIXTURE="$1"
			;;
		--fidelity-reference)
			shift
			if [ "$#" -eq 0 ]; then
				usage
				exit 2
			fi
			FIDELITY_REFERENCE="$1"
			;;
		--help|-h)
			usage
			exit 0
			;;
		*)
			usage
			exit 2
			;;
	esac
	shift
done

cd "$REPO_ROOT"

families=(pretrained ft_places2 ft_celebahq ft_ffhq)
precisions=(f32 f16 q8 q4 mxfp8 mxfp4)
quant_precisions=(q8 q4 mxfp8 mxfp4)

log() {
	printf '\n==> %s\n' "$1"
}

require_file() {
	if [ ! -f "$1" ]; then
		printf 'Missing file: %s\n' "$1" >&2
		exit 1
	fi
}

require_nonempty_file() {
	require_file "$1"
	if [ ! -s "$1" ]; then
		printf 'Empty file: %s\n' "$1" >&2
		exit 1
	fi
}

require_png() {
	require_nonempty_file "$1"
	python3 - "$1" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
if path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
    raise SystemExit(f"Not a PNG file: {path}")
PY
}

require_dir() {
	if [ ! -d "$1" ]; then
		printf 'Missing directory: %s\n' "$1" >&2
		exit 1
	fi
}

run_catalyst_autorun() {
	local app_path="$1"
	local precision="$2"
	local status_name="$3"
	shift 3
	local expected_parts=("$@")
	local status_file="/tmp/moebius-demo-${status_name}-status.txt"
	local log_file="/tmp/moebius-demo-${status_name}.log"
	local compare="0"
	if [ "$status_name" = "compare" ]; then
		compare="1"
	fi
	rm -f "$status_file" "$log_file"

	MOEBIUS_MLX_MODEL_DIR="$MODEL_ROOT" \
	MOEBIUS_MODEL_ROOT="$MODEL_ROOT" \
	MOEBIUS_DEMO_AUTORUN=1 \
	MOEBIUS_DEMO_COMPARE="$compare" \
	MOEBIUS_DEMO_MODEL=ft_places2 \
	MOEBIUS_DEMO_PRECISION="$precision" \
	MOEBIUS_DEMO_STEPS=2 \
	MOEBIUS_DEMO_AUTORUN_STATUS="$status_file" \
	"$app_path/Contents/MacOS/MoebiusDemo" >"$log_file" 2>&1 &
	local pid=$!
	local status=""
	for _ in $(seq 1 240); do
		if [ -f "$status_file" ]; then
			status="$(cat "$status_file")"
			case "$status" in
				done*|error*)
					break
					;;
			esac
		fi
		sleep 1
	done

	kill "$pid" >/dev/null 2>&1 || true
	wait "$pid" >/dev/null 2>&1 || true

	if [ -z "$status" ]; then
		printf 'Catalyst autorun did not write a final status. Log:\n' >&2
		cat "$log_file" >&2 || true
		exit 1
	fi
	if [[ "$status" != done* ]]; then
		printf 'Catalyst autorun failed: %s\n' "$status" >&2
		cat "$log_file" >&2 || true
		exit 1
	fi
	for expected in "${expected_parts[@]}"; do
		if [[ "$status" != *"$expected"* ]]; then
			printf 'Catalyst autorun status missing "%s": %s\n' "$expected" "$status" >&2
			exit 1
		fi
	done
	printf '%s\n' "$status"
}

log "Checking converted artifact directories"
require_dir "$MODEL_ROOT"

for family in "${families[@]}"; do
	for precision in "${precisions[@]}"; do
		variant="$family-$precision"
		variant_dir="$MODEL_ROOT/$variant"
		require_dir "$variant_dir"
		require_file "$variant_dir/manifest.json"
		require_file "$variant_dir/unet.safetensors"
		require_file "$variant_dir/vae_encoder.safetensors"
		require_file "$variant_dir/vae_decoder.safetensors"
	done
done
printf 'Found all 24 f32/f16/quantized model directories.\n'

log "Checking experimental candidate model views"
candidate_variants=(ft_places2-q4-ff-spatial-vae ft_places2-q4-all-spatial-vae)
for variant in "${candidate_variants[@]}"; do
	variant_dir="$MODEL_ROOT/$variant"
	require_dir "$variant_dir"
	require_file "$variant_dir/manifest.json"
	require_file "$variant_dir/vae_encoder_quantized.safetensors"
	require_file "$variant_dir/vae_decoder_quantized.safetensors"
	if ! grep -q '"vaeEncoderQuantized"' "$variant_dir/manifest.json"; then
		printf 'Candidate manifest does not reference vaeEncoderQuantized: %s\n' "$variant_dir/manifest.json" >&2
		exit 1
	fi
	if ! grep -q '"vaeDecoderQuantized"' "$variant_dir/manifest.json"; then
		printf 'Candidate manifest does not reference vaeDecoderQuantized: %s\n' "$variant_dir/manifest.json" >&2
		exit 1
	fi
done
require_file "$MODEL_ROOT/ft_places2-q4-ff-spatial-vae/unet_quantized_pointwise_feedforward_spatial.safetensors"
require_file "$MODEL_ROOT/ft_places2-q4-all-spatial-vae/unet_quantized_pointwise_all_spatial.safetensors"
printf 'Found ft_places2 q4 candidate views and quantized VAE files.\n'

log "Checking quantized UNet artifacts"
quantized_count=0
for family in "${families[@]}"; do
	for precision in "${quant_precisions[@]}"; do
		variant_dir="$MODEL_ROOT/$family-$precision"
		require_file "$variant_dir/unet_quantized.safetensors"
		if ! grep -q '"unetQuantized"' "$variant_dir/manifest.json"; then
			printf 'Manifest does not reference unetQuantized: %s\n' "$variant_dir/manifest.json" >&2
			exit 1
		fi
		quantized_count=$((quantized_count + 1))
	done
done
if [ "$quantized_count" != "16" ]; then
	printf 'Expected to validate 16 standard quantized UNet files, validated %s\n' "$quantized_count" >&2
	exit 1
fi
printf 'Found all 16 quantized UNet files and manifest references.\n'

log "Checking checked-in benchmark report artifacts"
benchmark_report_dir="Docs/BenchmarkReports/ft_places2-onnx-steps20-visible-variants"
require_dir "$benchmark_report_dir"
require_nonempty_file "$benchmark_report_dir/benchmark.tsv"
require_nonempty_file "$benchmark_report_dir/summary.md"
for chart in \
	fidelity_errors.png \
	performance_timings.png \
	model_size.png \
	speed_fidelity_tradeoff.png \
	size_fidelity_tradeoff.png
do
	require_png "$benchmark_report_dir/$chart"
done
if ! grep -q 'ft_places2-q4-all-spatial-vae' "$benchmark_report_dir/benchmark.tsv"; then
	printf 'Benchmark TSV does not include the 200 MB visible demo candidate.\n' >&2
	exit 1
fi
if ! grep -q '200.50' "$benchmark_report_dir/benchmark.tsv"; then
	printf 'Benchmark TSV does not include the expected 200.50 MB package size.\n' >&2
	exit 1
fi
printf 'Checked benchmark TSV, summary, and five PNG charts.\n'

log "Checking demo project cleanup and lifecycle wiring"
if [ -d "Demo/MoebiusDemo---old" ]; then
	printf 'Old broken demo project should have been removed: Demo/MoebiusDemo---old\n' >&2
	exit 1
fi
require_file "Demo/MoebiusDemo/MoebiusDemo/SceneDelegate.swift"
require_file "Demo/MoebiusDemo/MoebiusDemo/ViewController.swift"
require_file "Demo/MoebiusDemo/MoebiusDemo/Info.plist"
if find Demo -name 'Main.storyboard' -print -quit | grep -q .; then
	printf 'Demo should not contain a main storyboard.\n' >&2
	exit 1
fi
if grep -q 'UIMainStoryboardFile' Demo/MoebiusDemo/MoebiusDemo/Info.plist; then
	printf 'Info.plist should not reference UIMainStoryboardFile.\n' >&2
	exit 1
fi
if ! grep -q 'UISceneDelegateClassName' Demo/MoebiusDemo/MoebiusDemo/Info.plist; then
	printf 'Info.plist should configure a scene delegate.\n' >&2
	exit 1
fi
if ! grep -q 'window.rootViewController = ViewController()' Demo/MoebiusDemo/MoebiusDemo/SceneDelegate.swift; then
	printf 'SceneDelegate should install ViewController programmatically.\n' >&2
	exit 1
fi
printf 'Checked demo cleanup, SceneDelegate wiring, and storyboard-free main UI setup.\n'

log "Checking final documentation references"
require_nonempty_file "Docs/PROJECT.md"
require_nonempty_file "Docs/API.md"
require_nonempty_file "README.md"
for expected in \
	"MoebiusModelResolver" \
	"MoebiusPipeline" \
	"MoebiusModelDeploymentInfo" \
	"benchmark-fixture" \
	"run_device_autorun.sh"
do
	if ! grep -q "$expected" Docs/API.md; then
		printf 'Docs/API.md missing expected reference: %s\n' "$expected" >&2
		exit 1
	fi
done
if ! grep -q 'Docs/API.md' README.md; then
	printf 'README.md does not link Docs/API.md\n' >&2
	exit 1
fi
if ! grep -q 'Docs/API.md' Docs/PROJECT.md; then
	printf 'Docs/PROJECT.md does not link Docs/API.md\n' >&2
	exit 1
fi
printf 'Checked README, project writeup, and API/CLI reference.\n'

log "Checking helper script syntax"
bash -n \
	Scripts/build_mlx_metallib.sh \
	Scripts/stage_models_for_ios.sh \
	Scripts/verify_local.sh \
	Scripts/benchmark_fidelity.sh \
	Scripts/run_benchmark_report.sh \
	Scripts/benchmark_image.sh \
	Scripts/run_image_benchmark_report.sh \
	Scripts/run_device_autorun.sh
if ! Scripts/run_device_autorun.sh --help | grep -q -- '--compare'; then
	printf 'run_device_autorun.sh help should document --compare\n' >&2
	exit 1
fi
if ! Scripts/run_device_autorun.sh --help | grep -q -- '--dry-run'; then
	printf 'run_device_autorun.sh help should document --dry-run\n' >&2
	exit 1
fi
if ! Scripts/run_device_autorun.sh --help | grep -q -- '--wait-seconds'; then
	printf 'run_device_autorun.sh help should document --wait-seconds\n' >&2
	exit 1
fi
if ! Scripts/run_device_autorun.sh --help | grep -q -- '--poll-seconds'; then
	printf 'run_device_autorun.sh help should document --poll-seconds\n' >&2
	exit 1
fi
if ! Scripts/run_device_autorun.sh --help | grep -q -- '--autorun-timeout-seconds'; then
	printf 'run_device_autorun.sh help should document --autorun-timeout-seconds\n' >&2
	exit 1
fi
if ! Scripts/run_device_autorun.sh --help | grep -q -- '--launch-wait-seconds'; then
	printf 'run_device_autorun.sh help should document --launch-wait-seconds\n' >&2
	exit 1
fi
device_compare_plan="$(Scripts/run_device_autorun.sh --model ft_places2 --compare --wait-seconds 30 --poll-seconds 3 --autorun-timeout-seconds 120 --launch-wait-seconds 90 --dry-run)"
for expected in \
	'compare: 1' \
	'wait_seconds: 30' \
	'poll_seconds: 3' \
	'autorun_timeout_seconds: 120' \
	'launch_wait_seconds: 90' \
	'xcodebuild_destination: generic/platform=iOS' \
	'ft_places2-f32' \
	'ft_places2-f16' \
	'ft_places2-q8' \
	'ft_places2-q4' \
	'ft_places2-mxfp8' \
	'ft_places2-mxfp4' \
	'ft_places2-q4-ff-spatial-vae' \
	'ft_places2-q4-all-spatial-vae' \
	'"MOEBIUS_DEMO_AUTORUN_RUN_ID":"' \
	'"MOEBIUS_DEMO_COMPARE":"1"'
do
	if ! printf '%s\n' "$device_compare_plan" | grep -q "$expected"; then
		printf 'run_device_autorun.sh dry-run compare plan missing "%s"\n' "$expected" >&2
		exit 1
	fi
done

log "Running Swift tests"
swift test

log "Running Catalyst demo unit tests"
demo_unit_test_derived_data="/tmp/moebius-demo-unit-tests-derived-data"
rm -rf "$demo_unit_test_derived_data"
xcodebuild \
	-project Demo/MoebiusDemo/MoebiusDemo.xcodeproj \
	-scheme MoebiusDemo \
	-configuration Debug \
	-destination 'platform=macOS,variant=Mac Catalyst' \
	-derivedDataPath "$demo_unit_test_derived_data" \
	test
printf 'Catalyst demo unit tests passed.\n'

if [ "$RUN_ALL_INSPECTS" -eq 1 ]; then
	log "Inspecting all 24 variants"
	for family in "${families[@]}"; do
		for precision in "${precisions[@]}"; do
			swift run moebius-mlx inspect "$family-$precision" >/dev/null
			printf 'inspected %s-%s\n' "$family" "$precision"
		done
	done
else
	log "Inspecting representative variants"
	swift run moebius-mlx inspect ft_places2-f32 >/dev/null
	swift run moebius-mlx inspect ft_places2-f16 >/dev/null
	swift run moebius-mlx inspect ft_places2-q8 >/dev/null
	swift run moebius-mlx inspect ft_places2-q4 >/dev/null
	swift run moebius-mlx inspect ft_places2-q4-all-spatial-vae >/dev/null
	printf 'inspected ft_places2-f32, ft_places2-f16, ft_places2-q8, ft_places2-q4, ft_places2-q4-all-spatial-vae\n'
fi

inspect_q8_output="$(swift run moebius-mlx inspect ft_places2-q8)"
for expected in \
	'loaded tensor payload total mb: 600.05' \
	'loaded tensor storage bytes per MLX element: 2.036' \
	'float16: elements=289547932 approx=579.10 MB / 552.27 MiB (96.5%)' \
	'uint32: elements=5238784 approx=20.96 MB / 19.98 MiB (3.5%)'
do
	if ! printf '%s\n' "$inspect_q8_output" | grep -Fq "$expected"; then
		printf 'inspect ft_places2-q8 output missing "%s"\n' "$expected" >&2
		exit 1
	fi
done

log "Running MLX smoke checks"
swift run moebius-mlx verify-fixture ft_places2-f32
swift run moebius-mlx image-smoke ft_places2-f16 --steps 2
swift run moebius-mlx image-smoke ft_places2-q4 --steps 2
swift run moebius-mlx compare-image ft_places2 --steps 2
swift run moebius-mlx benchmark-image ft_places2-q4 \
	--reference-model ft_places2-f16 \
	--reference-name f16-image-verify \
	--steps 2 \
	--samples gradient-circle

log "Checking trimmed iPad staging package"
staging_dir="$(mktemp -d /tmp/moebius-stage-verify.XXXXXX)"
trap 'rm -rf "$staging_dir"' EXIT
Scripts/stage_models_for_ios.sh --output "$staging_dir" ft_places2-q4
require_file "$staging_dir/Moebius-MLX/ft_places2-q4/manifest.json"
require_file "$staging_dir/Moebius-MLX/ft_places2-q4/unet_quantized_pointwise_feedforward.safetensors"
require_file "$staging_dir/Moebius-MLX/ft_places2-q4/vae_encoder.safetensors"
require_file "$staging_dir/Moebius-MLX/ft_places2-q4/vae_decoder.safetensors"
if [ -f "$staging_dir/Moebius-MLX/ft_places2-q4/unet.safetensors" ]; then
	printf 'Trimmed quantized staging package should not include unet.safetensors\n' >&2
	exit 1
fi
swift run moebius-mlx inspect "$staging_dir/Moebius-MLX/ft_places2-q4" >/dev/null
printf 'trimmed ft_places2-q4 staging package loaded successfully\n'

Scripts/stage_models_for_ios.sh --output "$staging_dir" ft_places2-q4-all-spatial-vae
require_file "$staging_dir/Moebius-MLX/ft_places2-q4-all-spatial-vae/manifest.json"
require_file "$staging_dir/Moebius-MLX/ft_places2-q4-all-spatial-vae/unet_quantized_pointwise_all_spatial.safetensors"
require_file "$staging_dir/Moebius-MLX/ft_places2-q4-all-spatial-vae/vae_encoder_quantized.safetensors"
require_file "$staging_dir/Moebius-MLX/ft_places2-q4-all-spatial-vae/vae_decoder_quantized.safetensors"
if [ -f "$staging_dir/Moebius-MLX/ft_places2-q4-all-spatial-vae/unet.safetensors" ]; then
	printf 'Trimmed candidate staging package should not include unet.safetensors\n' >&2
	exit 1
fi
swift run moebius-mlx inspect "$staging_dir/Moebius-MLX/ft_places2-q4-all-spatial-vae" >/dev/null
printf 'trimmed ft_places2-q4-all-spatial-vae staging package loaded successfully\n'

if [ -n "$FIDELITY_FIXTURE" ]; then
	log "Running cross-runtime fidelity benchmark"
	Scripts/benchmark_fidelity.sh \
		--family ft_places2 \
		--reference "$FIDELITY_REFERENCE" \
		--fixture "$FIDELITY_FIXTURE" \
		--precisions f32,f16,q8,q4,mxfp8,mxfp4
fi

if [ "$RUN_DEMO_BUILDS" -eq 1 ]; then
	demo_derived_data="/tmp/moebius-demo-verify-derived-data"
	demo_project="$REPO_ROOT/Demo/MoebiusDemo/MoebiusDemo.xcodeproj"
	rm -rf "$demo_derived_data"

	log "Building demo for Mac Catalyst"
	xcodebuild \
		-project "$demo_project" \
		-scheme MoebiusDemo \
		-destination 'platform=macOS,variant=Mac Catalyst' \
		-derivedDataPath "$demo_derived_data" \
		build

	catalyst_app="$demo_derived_data/Build/Products/Debug-maccatalyst/MoebiusDemo.app"
	require_dir "$catalyst_app"

	log "Running Catalyst autorun size checks"
	run_catalyst_autorun "$catalyst_app" f16 f16 "ft_places2-f16" "seconds=" "model_mb=619.89" "memory_mb="
	run_catalyst_autorun "$catalyst_app" q4 q4 "ft_places2-q4" "seconds=" "model_mb=475.67" "memory_mb="
	run_catalyst_autorun "$catalyst_app" ft_places2-q4-all-spatial-vae q4-200 "ft_places2-q4-all-spatial-vae" "seconds=" "model_mb=200.50" "memory_mb="
	run_catalyst_autorun "$catalyst_app" f16 compare "done comparison 8/8" "ft_places2-q4-all-spatial-vae:200.50" "memory_mbs=" "seconds="

	log "Building demo for iOS Simulator"
	xcodebuild \
		-project "$demo_project" \
		-scheme MoebiusDemo \
		-destination 'generic/platform=iOS Simulator' \
		-derivedDataPath "$demo_derived_data" \
		build
fi

log "Verification complete"
