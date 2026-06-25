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
XCODEBUILD_DESTINATION="${XCODEBUILD_DESTINATION:-generic/platform=iOS}"
STAGE_DIR="${STAGE_DIR:-/tmp/moebius-ipad-autorun-stage}"
STATUS_OUT="${STATUS_OUT:-/tmp/moebius-ipad-autorun-status.txt}"
WAIT_SECONDS=0
POLL_SECONDS=5
AUTORUN_TIMEOUT_SECONDS=900
LAUNCH_WAIT_SECONDS=""
SKIP_BUILD=0
SKIP_INSTALL=0
SKIP_COPY=0
VARIANT_WAS_EXPLICIT=0
COMPARE=0
DRY_RUN=0

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
  --xcodebuild-destination DEST
                    xcodebuild destination. Default: $XCODEBUILD_DESTINATION
                    CoreDevice ids are still used for install/copy/launch.
  --stage-dir DIR    Temporary staging path. Default: $STAGE_DIR
  --status-out PATH  Host path for copied status. Default: $STATUS_OUT
  --wait-seconds N   Wait up to N seconds for the device. Default: $WAIT_SECONDS
  --poll-seconds N   Device/status poll interval. Default: $POLL_SECONDS
  --autorun-timeout-seconds N
                    Wait up to N seconds for done/error autorun status after
                    launch. Default: $AUTORUN_TIMEOUT_SECONDS
  --launch-wait-seconds N
                    Retry launch for N seconds if the device is locked.
                    Default: same as --wait-seconds
  --skip-build       Reuse existing device build.
  --skip-install     Reuse installed app.
  --skip-copy        Reuse model already copied to app container.
  --dry-run          Print resolved options and exit before device lookup.
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
		--xcodebuild-destination)
			XCODEBUILD_DESTINATION="$2"
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
		--wait-seconds)
			WAIT_SECONDS="$2"
			shift 2
			;;
		--poll-seconds)
			POLL_SECONDS="$2"
			shift 2
			;;
		--autorun-timeout-seconds)
			AUTORUN_TIMEOUT_SECONDS="$2"
			shift 2
			;;
		--launch-wait-seconds)
			LAUNCH_WAIT_SECONDS="$2"
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
		--dry-run)
			DRY_RUN=1
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

case "$WAIT_SECONDS" in
	''|*[!0-9]*)
		printf -- '--wait-seconds must be a non-negative integer: %s\n' "$WAIT_SECONDS" >&2
		exit 2
		;;
esac
case "$POLL_SECONDS" in
	''|*[!0-9]*)
		printf -- '--poll-seconds must be a positive integer: %s\n' "$POLL_SECONDS" >&2
		exit 2
		;;
esac
if [ "$POLL_SECONDS" -eq 0 ]; then
	printf -- '--poll-seconds must be greater than zero.\n' >&2
	exit 2
fi
case "$AUTORUN_TIMEOUT_SECONDS" in
	''|*[!0-9]*)
		printf -- '--autorun-timeout-seconds must be a non-negative integer: %s\n' "$AUTORUN_TIMEOUT_SECONDS" >&2
		exit 2
		;;
esac
if [ -z "$LAUNCH_WAIT_SECONDS" ]; then
	LAUNCH_WAIT_SECONDS="$WAIT_SECONDS"
fi
case "$LAUNCH_WAIT_SECONDS" in
	''|*[!0-9]*)
		printf -- '--launch-wait-seconds must be a non-negative integer: %s\n' "$LAUNCH_WAIT_SECONDS" >&2
		exit 2
		;;
esac

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

autorun_id="$(date +%Y%m%d%H%M%S)-$$"
launch_environment="{\"MOEBIUS_DEMO_AUTORUN\":\"1\",\"MOEBIUS_DEMO_AUTORUN_RUN_ID\":\"$autorun_id\",\"MOEBIUS_DEMO_COMPARE\":\"$COMPARE\",\"MOEBIUS_DEMO_MODEL\":\"$MODEL\",\"MOEBIUS_DEMO_PRECISION\":\"$DEMO_PRECISION\",\"MOEBIUS_DEMO_STEPS\":\"$STEPS\"}"

if [ "$DRY_RUN" -eq 1 ]; then
	printf 'device: %s\n' "$DEVICE_ID"
	printf 'bundle_id: %s\n' "$BUNDLE_ID"
	printf 'model: %s\n' "$MODEL"
	printf 'precision: %s\n' "$PRECISION"
	printf 'variant: %s\n' "$VARIANT"
	printf 'compare: %s\n' "$COMPARE"
	printf 'steps: %s\n' "$STEPS"
	printf 'derived_data: %s\n' "$DERIVED_DATA"
	printf 'xcodebuild_destination: %s\n' "$XCODEBUILD_DESTINATION"
	printf 'stage_dir: %s\n' "$STAGE_DIR"
	printf 'status_out: %s\n' "$STATUS_OUT"
	printf 'wait_seconds: %s\n' "$WAIT_SECONDS"
	printf 'poll_seconds: %s\n' "$POLL_SECONDS"
	printf 'autorun_timeout_seconds: %s\n' "$AUTORUN_TIMEOUT_SECONDS"
	printf 'launch_wait_seconds: %s\n' "$LAUNCH_WAIT_SECONDS"
	printf 'autorun_id: %s\n' "$autorun_id"
	printf 'stage_variants:\n'
	for stage_variant in "${stage_variants[@]}"; do
		printf '  %s\n' "$stage_variant"
	done
	printf 'launch_environment: %s\n' "$launch_environment"
	exit 0
fi

device_line=""
deadline="$(( $(date +%s) + WAIT_SECONDS ))"
while :; do
	device_line="$(xcrun devicectl list devices | awk -v id="$DEVICE_ID" '$0 ~ id { print; found=1 } END { if (!found) exit 1 }')" || {
		printf 'Device not found: %s\n' "$DEVICE_ID" >&2
		xcrun devicectl list devices >&2
		exit 1
	}
	case "$device_line" in
		*" available "*) break ;;
		*" unavailable "*)
			if [ "$WAIT_SECONDS" -eq 0 ] || [ "$(date +%s)" -ge "$deadline" ]; then
				printf 'Device is not available:\n%s\n\n' "$device_line" >&2
				printf 'Unlock/connect the device, then rerun this script.\n' >&2
				exit 1
			fi
			printf 'Waiting for device to become available: %s\n' "$device_line" >&2
			sleep "$POLL_SECONDS"
			;;
		*)
			printf 'Device is not in an available state:\n%s\n' "$device_line" >&2
			exit 1
			;;
	esac
done

if [ "$SKIP_BUILD" -eq 0 ]; then
	xcodebuild \
		-project "$ROOT_DIR/Demo/MoebiusDemo/MoebiusDemo.xcodeproj" \
		-scheme MoebiusDemo \
		-destination "$XCODEBUILD_DESTINATION" \
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

launch_output="$(mktemp /tmp/moebius-device-launch.XXXXXX)"
deadline="$(( $(date +%s) + LAUNCH_WAIT_SECONDS ))"
while :; do
	if xcrun devicectl device process launch \
		--device "$DEVICE_ID" \
		--terminate-existing \
		--environment-variables "$launch_environment" \
		"$BUNDLE_ID" >"$launch_output" 2>&1; then
		cat "$launch_output"
		rm -f "$launch_output"
		break
	fi
	if grep -qi 'Locked' "$launch_output" && [ "$LAUNCH_WAIT_SECONDS" -gt 0 ] && [ "$(date +%s)" -lt "$deadline" ]; then
		printf 'Device is locked; waiting before retrying launch.\n' >&2
		sleep "$POLL_SECONDS"
		continue
	fi
	cat "$launch_output" >&2
	rm -f "$launch_output"
	exit 1
done

rm -f "$STATUS_OUT"
status=""
deadline="$(( $(date +%s) + AUTORUN_TIMEOUT_SECONDS ))"
while :; do
	if xcrun devicectl device copy from \
		--device "$DEVICE_ID" \
		--source Documents/moebius-demo-autorun-status.txt \
		--destination "$STATUS_OUT" \
		--domain-type appDataContainer \
		--domain-identifier "$BUNDLE_ID" >/dev/null 2>&1; then
		status="$(cat "$STATUS_OUT")"
		run_prefix="run_id=$autorun_id "
		if [[ "$status" == "$run_prefix"* ]]; then
			status_body="${status#"$run_prefix"}"
			case "$status_body" in
				done*)
					break
					;;
				error*)
					printf 'Device autorun failed: %s\n' "$status" >&2
					exit 1
					;;
			esac
		fi
	fi
	if [ "$AUTORUN_TIMEOUT_SECONDS" -eq 0 ] || [ "$(date +%s)" -ge "$deadline" ]; then
		printf 'Timed out waiting for autorun status for run_id=%s.\n' "$autorun_id" >&2
		if [ -n "$status" ]; then
			printf 'Latest copied status: %s\n' "$status" >&2
		fi
		exit 1
	fi
	sleep "$POLL_SECONDS"
done

cat "$STATUS_OUT"
