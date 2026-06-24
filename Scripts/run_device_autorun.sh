#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE_ID="${MOEBIUS_DEVICE_ID:-4B1F3A7A-CB7A-5D9A-85FF-D7578C7EC4C1}"
BUNDLE_ID="${MOEBIUS_DEMO_BUNDLE_ID:-com.ikyle.MoebiusDemo}"
MODEL="${MOEBIUS_DEMO_MODEL:-ft_places2}"
PRECISION="${MOEBIUS_DEMO_PRECISION:-q4}"
STEPS="${MOEBIUS_DEMO_STEPS:-2}"
VARIANT="${MODEL}-${PRECISION}"
DERIVED_DATA="${DERIVED_DATA:-/tmp/MoebiusDemoDeviceDerivedData}"
STAGE_DIR="${STAGE_DIR:-/tmp/moebius-ipad-autorun-stage}"
STATUS_OUT="${STATUS_OUT:-/tmp/moebius-ipad-autorun-status.txt}"
SKIP_BUILD=0
SKIP_INSTALL=0
SKIP_COPY=0
VARIANT_WAS_EXPLICIT=0
COMPARE=0

usage() {
	cat <<USAGE
Usage: Scripts/run_device_autorun.sh [options]

Builds the UIKit demo for a physical iPad, installs it, stages model variants,
copies the staged Moebius-MLX folder into the app Documents container, launches
the app with MOEBIUS_DEMO_AUTORUN=1, and reads back the autorun status file.

Options:
  --device ID        CoreDevice id. Default: $DEVICE_ID
  --bundle-id ID     App bundle id. Default: $BUNDLE_ID
  --variant NAME     Full model variant, e.g. ft_places2-q4.
                    Overrides --model/--precision when supplied.
  --model NAME       Model family. Default: $MODEL
  --precision NAME   Precision/candidate suffix. Default: $PRECISION
  --compare          Run the demo comparison path. Stages all visible variants
                    for --model, including ft_places2 q4 size candidates.
  --steps N          Denoising steps. Default: $STEPS
  --derived-data DIR DerivedData path. Default: $DERIVED_DATA
  --stage-dir DIR    Temporary staging path. Default: $STAGE_DIR
  --status-out PATH  Host path for copied status. Default: $STATUS_OUT
  --skip-build       Reuse existing device build.
  --skip-install     Reuse installed app.
  --skip-copy        Reuse model already copied to app container.
USAGE
}

while [ "$#" -gt 0 ]; do
	case "$1" in
		--device)
			DEVICE_ID="$2"
			shift 2
			;;
		--bundle-id)
			BUNDLE_ID="$2"
			shift 2
			;;
		--variant)
			VARIANT="$2"
			VARIANT_WAS_EXPLICIT=1
			shift 2
			;;
		--model)
			MODEL="$2"
			VARIANT="${MODEL}-${PRECISION}"
			shift 2
			;;
		--precision)
			PRECISION="$2"
			VARIANT="${MODEL}-${PRECISION}"
			shift 2
			;;
		--compare)
			COMPARE=1
			shift
			;;
		--steps)
			STEPS="$2"
			shift 2
			;;
		--derived-data)
			DERIVED_DATA="$2"
			shift 2
			;;
		--stage-dir)
			STAGE_DIR="$2"
			shift 2
			;;
		--status-out)
			STATUS_OUT="$2"
			shift 2
			;;
		--skip-build)
			SKIP_BUILD=1
			shift
			;;
		--skip-install)
			SKIP_INSTALL=1
			shift
			;;
		--skip-copy)
			SKIP_COPY=1
			shift
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

if [ "$VARIANT_WAS_EXPLICIT" -eq 1 ]; then
	case "$VARIANT" in
		pretrained-*)
			MODEL="pretrained"
			PRECISION="${VARIANT#pretrained-}"
			;;
		ft_places2-*)
			MODEL="ft_places2"
			PRECISION="${VARIANT#ft_places2-}"
			;;
		ft_celebahq-*)
			MODEL="ft_celebahq"
			PRECISION="${VARIANT#ft_celebahq-}"
			;;
		ft_ffhq-*)
			MODEL="ft_ffhq"
			PRECISION="${VARIANT#ft_ffhq-}"
			;;
		*)
			printf 'Cannot infer model family from variant: %s\n' "$VARIANT" >&2
			exit 2
			;;
	esac
fi
case "$PRECISION" in
	f32|f16|q8|q4|mxfp8|mxfp4)
	DEMO_PRECISION="$PRECISION"
	;;
	*)
	DEMO_PRECISION="$VARIANT"
	;;
esac

stage_variants=()
if [ "$COMPARE" -eq 1 ]; then
	for precision in f32 f16 q8 q4 mxfp8 mxfp4; do
		stage_variants+=("$MODEL-$precision")
	done
	if [ "$MODEL" = "ft_places2" ]; then
		stage_variants+=("ft_places2-q4-ff-spatial-vae")
		stage_variants+=("ft_places2-q4-all-spatial-vae")
	fi
else
	stage_variants=("$VARIANT")
fi

device_line="$(xcrun devicectl list devices | awk -v id="$DEVICE_ID" '$0 ~ id { print; found=1 } END { if (!found) exit 1 }')" || {
	printf 'Device not found: %s\n' "$DEVICE_ID" >&2
	xcrun devicectl list devices >&2
	exit 1
}
case "$device_line" in
	*" unavailable "*)
		printf 'Device is not available:\n%s\n\n' "$device_line" >&2
		printf 'Unlock/connect the device, then rerun this script.\n' >&2
		exit 1
		;;
	*" available "*) ;;
	*)
		printf 'Device is not in an available state:\n%s\n' "$device_line" >&2
		exit 1
		;;
esac

if [ "$SKIP_BUILD" -eq 0 ]; then
	xcodebuild \
		-project "$ROOT_DIR/Demo/MoebiusDemo/MoebiusDemo.xcodeproj" \
		-scheme MoebiusDemo \
		-destination "platform=iOS,id=$DEVICE_ID" \
		-derivedDataPath "$DERIVED_DATA" \
		build
fi

app_path="$(find "$DERIVED_DATA/Build/Products" -path '*/Debug-iphoneos/MoebiusDemo.app' -print -quit)"
if [ -z "$app_path" ]; then
	printf 'Could not find built MoebiusDemo.app under %s\n' "$DERIVED_DATA" >&2
	exit 1
fi

if [ "$SKIP_INSTALL" -eq 0 ]; then
	xcrun devicectl device install app \
		--device "$DEVICE_ID" \
		"$app_path"
fi

if [ "$SKIP_COPY" -eq 0 ]; then
	"$ROOT_DIR/Scripts/stage_models_for_ios.sh" \
		--output "$STAGE_DIR" \
		"${stage_variants[@]}"
	xcrun devicectl device copy to \
		--device "$DEVICE_ID" \
		--source "$STAGE_DIR/Moebius-MLX" \
		--destination Documents \
		--domain-type appDataContainer \
		--domain-identifier "$BUNDLE_ID" \
		--remove-existing-content false
fi

xcrun devicectl device process launch \
	--device "$DEVICE_ID" \
	--terminate-existing \
	--environment-variables "{\"MOEBIUS_DEMO_AUTORUN\":\"1\",\"MOEBIUS_DEMO_COMPARE\":\"$COMPARE\",\"MOEBIUS_DEMO_MODEL\":\"$MODEL\",\"MOEBIUS_DEMO_PRECISION\":\"$DEMO_PRECISION\",\"MOEBIUS_DEMO_STEPS\":\"$STEPS\"}" \
	"$BUNDLE_ID"

xcrun devicectl device copy from \
	--device "$DEVICE_ID" \
	--source Documents/moebius-demo-autorun-status.txt \
	--destination "$STATUS_OUT" \
	--domain-type appDataContainer \
	--domain-identifier "$BUNDLE_ID"

cat "$STATUS_OUT"
