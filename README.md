# Moebius-MLX

Swift/MLX port of Moebius image inpainting for Apple Silicon macOS, Mac Catalyst, and iPadOS.

GitHub: `https://github.com/kylehowells/Moebius-MLX`

CI: `.github/workflows/ci.yml` runs Swift package tests, CLI help, script syntax checks, Python helper syntax checks, and checked-in benchmark report validation. Full model verification still uses `Scripts/verify_local.sh` locally because the converted model artifacts live outside the Git repo.

This repo contains:

- `MoebiusMLX`: Swift library target for loading converted Moebius artifacts and running inference.
- `moebius-mlx`: CLI target for inspecting models and smoke-testing runtime paths.
- `Demo/MoebiusDemo`: UIKit demo app for comparing model families and precision variants with either the built-in sample or an imported image.
- `Scripts/convert_moebius.py`: PyTorch-to-MLX artifact conversion script.
- `Scripts/generate_reference_fixture.py`: deterministic PyTorch/ONNX latent fixture generator for fidelity checks.
- `Scripts/benchmark_fidelity.sh`: wrapper that compares MLX f32/f16/quant variants against a generated reference fixture.
- `Scripts/run_benchmark_report.sh`: one-command benchmark report generator that writes TSV data, PNG graphs, and a markdown summary.
- `Scripts/benchmark_image.sh` and `Scripts/run_image_benchmark_report.sh`: image-output fidelity/performance benchmark wrappers for quantized candidates.
- `Scripts/materialize_candidate_variants.py`: creates stable manifest views for the experimental 322 MB and 200 MB `ft_places2` q4 candidates.
- `Docs/PROJECT.md`: implementation notes, verification results, decisions, and limitations.
- `Docs/API.md`: focused Swift API and CLI reference.

Converted model artifacts live outside the Git repo:

```text
/Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX
```

`MoebiusModelResolver` searches, in order:

- explicit `preferredDirectory` / `modelRoot` arguments
- a direct local path to a variant directory or to `manifest.json`
- `MOEBIUS_MLX_MODEL_DIR`, plus `MOEBIUS_MODEL_ROOT` as a demo/autorun compatibility alias
- app bundle resources: `<Resources>/<variant>`, `<Resources>/Moebius-MLX/<variant>`, or `<Resources>/Models/<variant>`
- app Application Support and Documents folders, directly or under `Moebius-MLX`
- the local development path above
- Hugging Face model repos when using `MoebiusModel.fromPretrained(...)` or `MoebiusModelResolver.resolveOrDownload(...)`

For app code, prefer the async downloader API when the model may not already be present:

```swift
let model = try await MoebiusModel.fromPretrained(
	"ft_places2-q4",
	onDownloadProgress: { progress in
		let fraction = progress.fractionCompleted
		let bytes = progress.completedUnitCount
		let total = progress.totalUnitCount
		// Update progress UI on the main actor.
	}
)
```

The same API also accepts a local directory or a direct path to `manifest.json`, so applications can bundle models, download them from their own server, or let the user pick an existing model folder without going through Hugging Face.

Named variants map to one Hugging Face repo per family/quant, for example `ft_places2-q4` maps to `iky1e/moebius-ft-places2-mlx-q4`. Passing a full repo id or Hugging Face URL also works:

```swift
let model = try await MoebiusModel.fromPretrained("iky1e/moebius-ft-places2-mlx-q4")
let sameModel = try await MoebiusModel.fromPretrained("https://huggingface.co/iky1e/moebius-ft-places2-mlx-q4")
```

On macOS, materialized Hub snapshots are written under `$HF_HUB_CACHE/moebius-mlx`, `$HF_HOME/hub/moebius-mlx`, or `~/.cache/huggingface/hub/moebius-mlx`. On iOS and Mac Catalyst sandboxed app builds, the default materialized snapshot base is app-specific Application Support: `Library/Application Support/Moebius-MLX/HuggingFace`. The underlying Hub content cache still uses the Swift Hugging Face default cache. Pass `downloadBase:` to override the materialized snapshot location.

## Converted Models

There are 24 standard converted artifact directories:

- model families: `pretrained`, `ft_places2`, `ft_celebahq`, `ft_ffhq`
- precisions: `f32`, `f16`, `q8`, `q4`, `mxfp8`, `mxfp4`

Example:

```text
/Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX/ft_places2-f16
```

Each directory contains a `manifest.json`, UNet weights, VAE encoder weights, and VAE decoder weights. Quantized variants also contain a materialized `unet_quantized.safetensors` file referenced by the manifest.

`ft_places2` also has two experimental q4 candidate views:

- `ft_places2-q4-ff-spatial-vae`: 322.50 MB audited deployment payload, quantized VAE plus feed-forward/spatial UNet packing.
- `ft_places2-q4-all-spatial-vae`: 200.28 MB audited deployment payload, the size-target boundary candidate with a larger measured fidelity drop.

Refresh those candidate directories with:

```bash
Scripts/materialize_candidate_variants.py
```

## Build

```bash
swift build
swift test
```

Run the local verification bundle:

```bash
Scripts/verify_local.sh
```

By default this checks the converted artifact directory shape, the two stable `ft_places2` q4 candidate views, quantized UNet files and manifest references, checked-in benchmark graphs, demo cleanup and storyboard-free `SceneDelegate` wiring, script syntax, Swift tests, representative model loads, the f32 fixture, f16/q4 image smoke tests, `compare-image ft_places2 --steps 2`, `benchmark-image ft_places2-q4` against f16, and trimmed iPad staging for both the default q4 package and the 191M size-target candidate.

For heavier checks:

```bash
Scripts/verify_local.sh --all-inspects --demo-builds
```

Run cross-runtime fidelity checks after generating or supplying a PyTorch/ONNX fixture:

```bash
Scripts/generate_reference_fixture.py \
  --model ft_places2 \
  --reference onnx \
  --steps 20 \
  --output /Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX-benchmarks/fixtures/ft_places2-onnx-steps20

Scripts/benchmark_fidelity.sh \
  --family ft_places2 \
  --reference onnx \
  --fixture /Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX-benchmarks/fixtures/ft_places2-onnx-steps20
```

`generate_reference_fixture.py` needs the Python reference stack for the selected reference. The generated report in this repo was created from an ONNX-backed fixture already present under the model workspace. Regenerating that fixture on this machine's default Python environment still needs `onnxruntime`; regenerating PyTorch references also needs `diffusers`.

Generate graphs and a markdown report from the same benchmark:

```bash
Scripts/run_benchmark_report.sh \
  --family ft_places2 \
  --reference onnx \
  --fixture /Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX-benchmarks/fixtures/ft_places2-onnx-steps20 \
  --output-dir Docs/BenchmarkReports/ft_places2-onnx-steps20-visible-variants
```

The report writes:

- `benchmark.tsv`
- `summary.md`
- `fidelity_errors.png`
- `performance_timings.png`
- `model_size.png`
- `speed_fidelity_tradeoff.png`
- `size_fidelity_tradeoff.png`

## CLI

Inspect a converted model:

```bash
swift run moebius-mlx inspect ft_places2-f16
```

`inspect` also prints the manifest-selected deployment files, loaded dtype payloads, and dtype byte share, which is the quickest way to understand whether a variant is genuinely compact or mostly f16:

```bash
swift run moebius-mlx inspect ft_places2-q8
swift run moebius-mlx inspect ft_places2-q4
swift run moebius-mlx inspect ft_places2-q4-all-spatial-vae
```

Current key outputs:

```text
ft_places2-q8:
deployment mb: 600.23
loaded tensor payload total mb: 600.05
loaded tensor payload by dtype:
  float16: approx=579.10 MB (96.5%)
  uint32: approx=20.96 MB (3.5%)

ft_places2-q4:
deployment mb: 475.67
loaded tensor payload total mb: 475.49
loaded tensor payload by dtype:
  float16: approx=425.34 MB (89.5%)
  uint32: approx=50.16 MB (10.5%)

ft_places2-q4-all-spatial-vae:
deployment mb: 200.50
loaded tensor payload total mb: 200.28
loaded tensor payload by dtype:
  uint32: approx=145.89 MB (72.8%)
  float16: approx=54.39 MB (27.2%)
```

Run the zero-latent UNet fixture:

```bash
swift run moebius-mlx verify-fixture ft_places2-f32
```

Run the full synthetic image path:

```bash
swift run moebius-mlx image-smoke ft_places2-f16 --steps 2
```

Compare quantized variants against the first listed precision on the same synthetic sample:

```bash
swift run moebius-mlx compare-image ft_places2 --steps 2
swift run moebius-mlx compare-image ft_places2 --precisions f16,q4 --steps 2
```

Run the image-output benchmark report used for the q4 candidate comparison:

```bash
Scripts/run_image_benchmark_report.sh \
  --family ft_places2 \
  --reference-model ft_places2-f16 \
  --reference-label f16-image-all-samples \
  --samples all \
  --variants 'ft_places2-q4-default=ft_places2-q4,ft_places2-q4-ff-spatial-vae=ft_places2-q4-ff-spatial-vae,ft_places2-q4-all-spatial-vae=ft_places2-q4-all-spatial-vae' \
  --steps 2 \
  --output-dir /Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX-benchmarks/reports/ft_places2-image-f16-steps2-q4-candidates-all-samples
```

For file-backed image/mask checks, pass `--image` and `--mask`. The CLI center-crops/resizes the image to 512x512 and thresholds white mask pixels as the inpaint region:

```bash
Scripts/run_image_benchmark_report.sh \
  --family ft_places2 \
  --reference-model ft_places2-f16 \
  --reference-label f16-real-smoke \
  --image /path/to/image.png \
  --mask /path/to/mask.png \
  --variants 'ft_places2-q4-default=ft_places2-q4,ft_places2-q4-all-spatial-vae=ft_places2-q4-all-spatial-vae' \
  --steps 2
```

Compare an MLX model to a deterministic PyTorch/ONNX latent fixture:

```bash
swift run moebius-mlx benchmark-fixture ft_places2-f32 \
  --fixture /Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX-benchmarks/fixtures/ft_places2-onnx-steps20 \
  --reference-name onnx
```

The benchmark reports wall time, artifact size, `max_abs`, `mean_abs`, `rmse`, `p99_abs`, `psnr_db`, and `cosine_similarity` against `final_latents.bin`. Use f32 as the strict port-parity target, then compare f16 and quantized variants against the same fixture to decide whether the size reduction is acceptable.

`benchmark-fixture` reports both `load_seconds` and `inference_seconds`. Use `inference_seconds` for quant speed comparisons and `total_seconds` when judging the demo/app experience.

Audit model size composition:

```bash
./.venv/bin/python Scripts/audit_model_size.py \
  /Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX/ft_places2-q4 \
  --deployment
```

Compare several runtime payloads in one compact table:

```bash
./.venv/bin/python Scripts/audit_model_size.py --deployment \
  /Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX/ft_places2-f16 \
  /Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX/ft_places2-q8 \
  /Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX/ft_places2-q4 \
  /Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX/ft_places2-q4-all-spatial-vae
```

For physical iPad staging, quantized packages are trimmed by default:

```bash
Scripts/stage_models_for_ios.sh \
  --output /tmp/moebius-ipad-models \
  ft_places2-q4
```

This copies `manifest.json`, `unet_quantized.safetensors`, `vae_encoder.safetensors`, and `vae_decoder.safetensors`. It deliberately omits the original `unet.safetensors` because the loader uses `unetQuantized` when present. Pass `--include-source-unet` only for debugging/source preservation.

Materialize or refresh a quantized UNet file for a quantized variant:

```bash
swift run moebius-mlx export-quantized ft_places2-q4
swift run moebius-mlx export-quantized ft_places2-q4 \
  --filename unet_quantized_pointwise_feedforward.safetensors \
  --pointwise-convolution-policy feedforward
```

Build the demo app for Catalyst:

```bash
xcodebuild -project Demo/MoebiusDemo/MoebiusDemo.xcodeproj \
  -scheme MoebiusDemo \
  -destination 'platform=macOS,variant=Mac Catalyst' \
  build
```

In the demo, `ft_places2` exposes the standard `f32`, `f16`, `q8`, `q4`, `mxfp8`, and `mxfp4` options plus `q4 322M` and `q4 200M` candidate options. The Compare button runs every visible option for the selected model family, so selecting `ft_places2` compares the standard variants and both package-size candidates. Recent-run rows show status, steps, runtime, output mean, manifest-selected deployment MB, and observed resident memory MB for each completed variant.

The demo UI is grouped with on-screen labels for model fine-tune, precision/quant selection, generation settings, mask tools, and image/run actions. It supports loading a source image, using the built-in sample, painting or erasing a blue inpaint mask directly over the image, clearing the mask, changing brush size, adjusting steps/guidance/seed, toggling paste-back, running one selected variant, or comparing the visible quant variants. Imported images start with an empty editable mask; the built-in sample starts with a circular mask so the app remains immediately runnable.

Build the demo app for iOS Simulator:

```bash
xcodebuild -project Demo/MoebiusDemo/MoebiusDemo.xcodeproj \
  -scheme MoebiusDemo \
  -destination 'generic/platform=iOS Simulator' \
  build
```

## iPad Model Provisioning

The demo does not bundle the multi-GB converted model artifacts by default. For a physical iPad, stage the variants you want to test:

```bash
Scripts/stage_models_for_ios.sh \
  --output /tmp/moebius-ipad-models \
  ft_places2-f16 ft_places2-q4
```

This creates:

```text
/tmp/moebius-ipad-models/Moebius-MLX/<variant>/manifest.json
```

Copy the `Moebius-MLX` folder into the installed `MoebiusDemo` app's Documents container through Finder file sharing. The demo app has `UIFileSharingEnabled` and `LSSupportsOpeningDocumentsInPlace` enabled, and `MoebiusModelResolver` searches both `Documents/Moebius-MLX/<variant>` and `Documents/<variant>`.

For `devicectl`, this command copies one staged model folder into the app data container:

```bash
xcrun devicectl device copy to \
  --device <coredevice-id> \
  --source /tmp/moebius-ipad-models/Moebius-MLX \
  --destination Documents \
  --domain-type appDataContainer \
  --domain-identifier com.ikyle.MoebiusDemo
```

When launched with `MOEBIUS_DEMO_AUTORUN=1`, the demo writes its latest status to `Documents/moebius-demo-autorun-status.txt` unless `MOEBIUS_DEMO_AUTORUN_STATUS` points to a different path. Completed single-variant status lines include `seconds=`, `model_mb=`, and `memory_mb=`. Completed comparison status lines include `model_mbs=`, `memory_mbs=`, and per-variant `seconds=` maps, so a physical-device autorun can capture timing and memory evidence through one status readback.

The repeatable physical-device autorun helper wraps build, install, staging, model copy, launch, and status readback:

```bash
Scripts/run_device_autorun.sh --variant ft_places2-q4
Scripts/run_device_autorun.sh --variant ft_places2-q4-all-spatial-vae
Scripts/run_device_autorun.sh --model ft_places2 --compare
Scripts/run_device_autorun.sh --model ft_places2 --compare --dry-run
Scripts/run_device_autorun.sh --model ft_places2 --compare --wait-seconds 300 --launch-wait-seconds 300 --autorun-timeout-seconds 900
```

It defaults to Kyle's iPad Pro CoreDevice id and exits before build/install if `devicectl` reports the device as unavailable. `--wait-seconds` changes that into a bounded poll, so the command can be started before the iPad is connected. `--launch-wait-seconds` retries the app launch if SpringBoard rejects it because the iPad is locked. The helper builds with `generic/platform=iOS` by default because CoreDevice ids are not the same ids shown by `xcodebuild -showdestinations`; the CoreDevice id is still used for install, model copy, launch, and status readback. After launch, the helper tags each autorun with `MOEBIUS_DEMO_AUTORUN_RUN_ID` and polls the app-container status file until that same run reports `done` or `error`, bounded by `--autorun-timeout-seconds`. In compare mode, it stages the six standard visible precisions for the selected model; for `ft_places2`, it also stages the `q4 322M` and `q4 200M` candidate packages shown in the demo. `--dry-run` prints the resolved variant list and launch environment without touching CoreDevice, which is useful when checking the next physical-device run while the iPad is offline.

## Library Usage

```swift
let directory = try MoebiusModelResolver.resolve(name: "ft_places2-f16")
let model = try MoebiusModel.load(from: directory)
let pipeline = MoebiusPipeline(model: model)

let input = MoebiusImageInput(imageNHWC: imageFloats, maskNHWC: maskFloats)
let options = MoebiusGenerationOptions(
	steps: 8,
	guidanceScale: 2,
	seed: 7,
	strength: 0.99,
	pasteBack: false
)
let rgb01 = try pipeline.runImageRGBFloats(input: input, options: options)
```

`imageFloats` are NHWC RGB values in `[-1, 1]`. `maskFloats` are NHWC single-channel values where `1` is the region to inpaint. `runImageRGBFloats` returns NHWC RGB values clipped to `[0, 1]`.

## Current Verification

Passed locally:

- `swift build`
- `swift test`
- all 24 converted model variants load through `moebius-mlx inspect`
- all 16 quantized variants have `unet_quantized.safetensors` referenced by their manifests
- `moebius-mlx verify-fixture ft_places2-f32`
- `moebius-mlx image-smoke ft_places2-f16 --steps 2`
- `moebius-mlx image-smoke ft_places2-q4 --steps 2`
- `moebius-mlx compare-image ft_places2 --steps 2`
- `Scripts/verify_local.sh --demo-builds`
- `Scripts/verify_local.sh` now passes with candidate view checks, helper script syntax checks, default image benchmark, and trimmed staging for `ft_places2-q4` and `ft_places2-q4-all-spatial-vae`
- `Scripts/stage_models_for_ios.sh --output /tmp/moebius-ios-stage-test ft_places2-q4`
- trimmed staging now omits `unet.safetensors` for quantized variants; staged `ft_places2-q4` currently uses the feed-forward pointwise q4 artifact, is about `454 MiB`, and loads directly through `moebius-mlx inspect`
- `Scripts/stage_models_for_ios.sh --output /tmp/moebius-ios-candidate-stage ft_places2-q4-all-spatial-vae` produced a trimmed `191M` package for the 200 MB candidate
- Catalyst demo build
- iOS Simulator demo build
- physical iPad build for `Kyle‘s iPad Pro (18.6)` / `00008103-001460290E31001E`
- `devicectl` install to physical iPad CoreDevice id `4B1F3A7A-CB7A-5D9A-85FF-D7578C7EC4C1`
- `devicectl` copied staged `ft_places2-q4` into the app Documents container; readback confirmed `Documents/ft_places2-q4/manifest.json`
- latest physical-device check on 2026-06-24 could not continue because `xcrun devicectl list devices` reported Kyle's iPad Pro (`4B1F3A7A-CB7A-5D9A-85FF-D7578C7EC4C1`) as `unavailable`; the current feed-forward q4 package is staged at `/tmp/moebius-ipad-feedforward-stage/Moebius-MLX/ft_places2-q4` and is about `454M`
- `Scripts/run_device_autorun.sh --variant ft_places2-q4-all-spatial-vae --skip-build --skip-install --skip-copy` was tested against the current unavailable CoreDevice state and correctly exits before build/install with the device status
- On 2026-06-25 the physical iPad was available again. `Scripts/run_device_autorun.sh --variant ft_places2-q4-all-spatial-vae --steps 2 --wait-seconds 30 --poll-seconds 5 --autorun-timeout-seconds 900` built the app with `generic/platform=iOS`, installed `com.ikyle.MoebiusDemo`, staged `ft_places2-q4-all-spatial-vae (191M)`, and copied it into the app Documents container. Launch/status readback still could not complete because SpringBoard rejected launch while the iPad was locked. The resume command is:
  `Scripts/run_device_autorun.sh --variant ft_places2-q4-all-spatial-vae --steps 2 --wait-seconds 10 --launch-wait-seconds 300 --poll-seconds 5 --autorun-timeout-seconds 900 --skip-build --skip-install --skip-copy`
- `Scripts/verify_local.sh` passed again on 2026-06-25 after the demo mask-editor and documentation updates, including docs checks, Swift tests, MLX image smoke checks, benchmark-image, and trimmed staging/load checks for `ft_places2-q4` and `ft_places2-q4-all-spatial-vae`
- Catalyst UI was rebuilt and visually checked on 2026-06-25 with the labeled sections and editable mask workflow; Clear mask, Paint, and Erase all responded in the running Mac Catalyst app
- `Scripts/verify_local.sh` passed again on 2026-06-25 after adding explicit demo headers/control labels and the Catalyst demo unit-test gate. That run covered artifact checks, benchmark chart checks, docs checks, Swift tests, isolated Catalyst demo unit tests, representative model loads, MLX image smoke checks, the f16-referenced q4 benchmark, and trimmed staging/load checks for `ft_places2-q4` and `ft_places2-q4-all-spatial-vae`
- Catalyst demo autorun through `ViewController` completed `ft_places2-f16` with two steps and wrote a status containing `seconds=`, `model_mb=619.89`, `memory_mb=`, and bounded image stats
- Catalyst demo autorun through `ViewController` completed the current feed-forward pointwise `ft_places2-q4` artifact with two steps and wrote:
  `done ft_places2-q4 steps=2 seconds=... model_mb=475.67 memory_mb=... min=0.0 max=1.0 mean=...`
- Catalyst demo compare autorun through `ViewController` completed all eight visible `ft_places2` precision/candidate variants with two steps and wrote:
  `done comparison 8/8 variants errors=0 model_mbs=... memory_mbs=... seconds=...`
- Catalyst demo autorun through `ViewController` completed the `q4 200M` candidate selector path with two steps and wrote:
  `done ft_places2-q4-all-spatial-vae steps=2 seconds=... model_mb=200.50 memory_mb=... min=0.0 max=1.0 mean=...`
- `moebius-mlx benchmark-fixture` consumes deterministic PyTorch/ONNX latent fixtures and reports fidelity metrics plus total model size. The command has been smoke-tested with a synthetic fixture and used for the checked-in ONNX-backed report. Regenerating references from scratch requires the Python reference dependencies listed above.
- `Scripts/run_benchmark_report.sh --fixture /tmp/moebius-empty-fixture --reference synthetic-zero --steps 2 --output-dir /tmp/moebius-benchmark-report-smoke` produced all five graph PNGs plus `benchmark.tsv` and `summary.md`. This validates the report pipeline, not real ONNX/PyTorch parity.
- Real ONNX-backed 20-step `ft_places2` fixture:
  `/Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX-benchmarks/fixtures/ft_places2-onnx-steps20`
- Repo-local ONNX-backed 20-step report:
  `Docs/BenchmarkReports/ft_places2-onnx-steps20-visible-variants`
- iPad Simulator demo launch shows the comparison UI from the programmatic `SceneDelegate` root controller, including labeled model tabs, precision tabs, generation settings, mask tools, Run sample, Compare quants, and Sample/Paint mask/Result panels
- iPad Simulator demo UI includes Load image, Use sample, Clear mask, brush-size, Paint/Erase, guidance, seed, and paste-back controls; imported images are center-cropped/resized to the 512x512 Moebius tensor size and start with an empty editable mask
- iPad Simulator UI snapshot shows the run-history comparison panel with recent runs and package MB; tapping Compare quants records f32, f16, q8, q4, mxfp8, and mxfp4 rows for standard model families, and the `ft_places2` UI also exposes `q4 322M` and `q4 200M`
- iPad Simulator autorun fails gracefully with:
  `error ft_places2-f16 iOS Simulator cannot run MLX models`

Recorded f32 UNet zero-fixture parity against ONNX:

- ONNX min/max/mean: `-0.3489277363`, `0.1178752705`, `-0.0739606619`
- MLX min/max/mean: `-0.3489233255`, `0.1178750768`, `-0.0739595667`
- max absolute error: about `2.08e-5`
- mean absolute error: about `1.88e-6`

Recorded `compare-image ft_places2 --steps 2` output:

```text
variant          mean        max_abs_vs_ref  mean_abs_vs_ref
ft_places2-f16   0.50244707  -               -
ft_places2-q8    0.5024465   0.00094777346   0.000006535905
ft_places2-q4    0.50244474  0.0038945675    0.000024847031
ft_places2-mxfp8 0.5024485   0.0074070096    0.000042056134
ft_places2-mxfp4 0.5024302   0.010975122     0.00005909382
```

Fidelity benchmark design:

- Generate a deterministic reference with fixed input image, fixed binary mask, fixed seed, fixed scheduler, fixed guidance, fixed strength, and fixed denoising step count.
- Compare latent tensors before image decode first. This isolates UNet/scheduler fidelity and avoids hiding errors behind VAE decode and paste-back.
- Metrics to track: `max_abs` catches worst-case layer/layout mistakes; `mean_abs` and `rmse` capture average drift; `p99_abs` catches broad tail degradation; `psnr_db` is easier to compare across runs; `cosine_similarity` catches directional agreement in latent space; total artifact MB measures whether the variant is approaching the iPad size target.
- Performance metrics to track: `load_seconds` for model startup cost, `inference_seconds` for the fixed latent denoising workload, and `total_seconds` for app-facing cold-run cost.
- Acceptance should be tiered: f32 should be near ONNX/PyTorch floating-point error; f16 should be small numeric drift; quantized variants should be judged by both latent metrics and visual/manual samples because lower-bit quantization can preserve average metrics while degrading local structure.
- The default quantized package is still about the VAE size plus a quantized subset of the UNet. Experimental VAE + broader UNet convolution packing can reach about 200 MB, but the current 200 MB candidate has substantial latent drift and noticeably larger two-step image error versus f16.

Recorded real ONNX-backed `ft_places2` 20-step latent benchmark for the demo-visible variants:

```text
variant          model MB  RMSE       cosine      inference s
PyTorch vs ONNX  n/a       0.003229   0.99999213  CPU reference
ft_places2-f32   1239.58   0.000006   1.00000000  26.777
ft_places2-f16   619.89    0.165905   0.98042941  20.783
ft_places2-q8    600.23    0.165784   0.98044908  20.021
ft_places2-q4    475.67    0.176075   0.97774184  18.039
ft_places2-mxfp8 599.56    0.164743   0.98046005  21.273
ft_places2-mxfp4 589.09    0.184187   0.97599477  20.942
ft_places2-q4-ff-spatial-vae  322.50  0.178230  0.97694558  17.184
ft_places2-q4-all-spatial-vae 200.50  0.254395  0.95212305  17.438
```

Repo-local graph report:

- [summary.md](Docs/BenchmarkReports/ft_places2-onnx-steps20-visible-variants/summary.md)
- [fidelity_errors.png](Docs/BenchmarkReports/ft_places2-onnx-steps20-visible-variants/fidelity_errors.png)
- [performance_timings.png](Docs/BenchmarkReports/ft_places2-onnx-steps20-visible-variants/performance_timings.png)
- [model_size.png](Docs/BenchmarkReports/ft_places2-onnx-steps20-visible-variants/model_size.png)
- [speed_fidelity_tradeoff.png](Docs/BenchmarkReports/ft_places2-onnx-steps20-visible-variants/speed_fidelity_tradeoff.png)
- [size_fidelity_tradeoff.png](Docs/BenchmarkReports/ft_places2-onnx-steps20-visible-variants/size_fidelity_tradeoff.png)

Interpretation: ONNX is close to the PyTorch reference over the same 20-step latent fixture, and MLX f32 is even tighter to ONNX. The current f16 and quantized paths buy size and some speed but accumulate meaningful latent drift; they should not be considered quality-equivalent until visual evaluation and potentially mixed-precision/quantization changes improve that curve. The current `ft_places2-q4` manifest points at the feed-forward pointwise policy artifact, cutting the runtime payload to 475.67 MB decimal / about 454 MiB staged. The `q4 322M` candidate adds spatial and VAE quantized files and is the fastest row in the current run, but with slightly worse latent drift. The `q4 200M` candidate hits the requested package-size target, but it has the largest falloff in this benchmark.

An additional experimental artifact, `unet_quantized_pointwise_feedforward_spatial.safetensors`, packs the feed-forward 1x1 convolutions plus eligible spatial down/up-sampler 3x3 convolutions. It trims the runtime payload to 442.56 MB decimal / about 422 MiB staged, but the focused 20-step ONNX fixture shows slightly worse fidelity and speed than feed-forward-only q4:

```text
variant                    model MB  RMSE      cosine      inference s
ft_places2-q4-ff-pw        475.67    0.176075  0.97774184  14.363
ft_places2-q4-ff-spatial   442.56    0.178230  0.97694558  15.436
```

The focused graph report is in:

```text
/Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX-benchmarks/reports/ft_places2-onnx-steps20-q4-spatial
```

The next experimental candidate also uses `vae_encoder_quantized.safetensors` and `vae_decoder_quantized.safetensors`, packing eligible VAE `Linear` and Conv2d weights. The latent fixture does not exercise VAE image reconstruction quality, so these RMSE/cosine numbers match the UNet-only spatial candidate; the useful evidence here is load/runtime plus the package-size drop. A two-step `image-smoke` run through the quantized VAE path completed with bounded `[0, 1]` output.

```text
variant                         model MB  RMSE      cosine      inference s
ft_places2-q4-ff-pw             475.67    0.176075  0.97774184  15.373
ft_places2-q4-ff-spatial        442.56    0.178230  0.97694558  15.244
ft_places2-q4-ff-spatial-vae    322.50    0.178230  0.97694558  15.267
```

The focused q4 VAE graph report is in:

```text
/Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX-benchmarks/reports/ft_places2-onnx-steps20-q4-vae
```

The size-target boundary artifact, `unet_quantized_pointwise_all_spatial.safetensors` plus the quantized VAE files, reaches the requested package size but has a large fidelity falloff in the 20-step latent benchmark:

```text
variant                            model MB  RMSE      cosine      inference s
ft_places2-q4-ff-spatial-vae       322.50    0.178230  0.97694558  12.981
ft_places2-q4-all-spatial-vae      200.50    0.254395  0.95212305  12.832
```

The audited deployment payload is 200.28 MB decimal / 191.00 MiB. It loads and completes `image-smoke --steps 2`, but this should be considered a boundary experiment until visual/image-level fidelity metrics show whether the quality loss is acceptable.

The focused 200 MB q4 graph report is in:

```text
/Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX-benchmarks/reports/ft_places2-onnx-steps20-q4-200mb
```

Image-level benchmark against `ft_places2-f16` on the deterministic synthetic image/mask/seed with two denoising steps:

```text
variant                         model MB  image RMSE  p99 abs   cosine      inference s
ft_places2-q4-default           475.67    0.000286    0.001356  0.99999988  1.271
ft_places2-q4-ff-spatial-vae    322.50    0.011555    0.058204  0.99979931  8.985
ft_places2-q4-all-spatial-vae   200.50    0.011488    0.057879  0.99980158  1.913
```

This confirms that VAE quantization is the dominant image-output change in these candidates: the 322 MB and 200 MB variants have similar two-step image error, while default q4 remains very close to f16. The report is in:

```text
/Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX-benchmarks/reports/ft_places2-image-f16-steps2-q4-candidates
```

The same image benchmark now also supports deterministic multi-sample aggregation with `--samples all`, covering gradient/circle, checker/square, radial/band, and diagonal/stripe profiles. The all-samples two-step aggregate is:

```text
variant                         model MB  image RMSE  p99 abs   cosine      inference s
ft_places2-q4-default           475.67    0.000343    0.001569  0.99999982  5.038
ft_places2-q4-ff-spatial-vae    322.50    0.014153    0.068663  0.99971294  15.224
ft_places2-q4-all-spatial-vae   200.50    0.014082    0.068348  0.99971575  6.968
```

This broader check preserves the earlier conclusion: the default q4 package remains the best quality/speed choice; the 200 MB package is real but has a much larger RGB error. The all-samples graph report is in:

```text
/Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX-benchmarks/reports/ft_places2-image-f16-steps2-q4-candidates-all-samples
```

A file-backed image/mask smoke benchmark through the same report path produced:

```text
variant                         model MB  image RMSE  p99 abs   cosine      inference s
ft_places2-q4-default           475.67    0.000260    0.001245  0.99999994  1.264
ft_places2-q4-all-spatial-vae   200.50    0.011295    0.057014  0.99983478  1.834
```

The real-image smoke graph report is in:

```text
/Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX-benchmarks/reports/ft_places2-image-f16-steps2-real-smoke
```

The current q4-policy graph report is in:

```text
/Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX-benchmarks/reports/ft_places2-onnx-steps20-q4-policies
```

Size audit for the current trimmed `ft_places2-q4` feed-forward pointwise deployment payload:

```text
total:                  453.47 MiB
UNet quantized file:    293.91 MiB
VAE decoder:             94.40 MiB
VAE encoder:             65.16 MiB
UNet 1x1 conv/proj:     165.58 MiB
UNet 3x3 conv/proj:      45.40 MiB
VAE conv weights:       155.48 MiB
actual uint32 packed:    47.83 MiB
```

Projected q4 payloads with no extra metadata:

```text
current runtime payload:       453.47 MiB
q4 UNet 1x1 conv only:         329.28 MiB
q4 all UNet conv/proj:         261.62 MiB
q4 UNet conv + relpos:         241.11 MiB
q4 UNet + VAE conv weights:    124.51 MiB
ideal all tensors q4:          107.39 MiB
```

This explains why current q4 is still large: the original q4 file only packs the MLX-quantized Linear/Embedding modules; almost all Conv2d and VAE weights remain f16. The same caveat applies to the current q8/mxfp8 artifacts: `ft_places2-q8` deploys about 572 MiB because it is a mostly-f16 pipeline bundle with a partially quantized UNet plus f16 VAE files, not a raw one-byte-per-parameter fp8 export. MLX Swift exposes quantized 2D matrix packing/matmul and `QuantizedLinear`/`QuantizedEmbedding`, but no built-in `QuantizedConv2d`. The package now includes `MoebiusQuantizedPointwiseConv2d`, a packed NHWC convolution backed by im2col + `quantizedMM`, with tests covering 1x1 and 3x3 primitives, explicit opt-in UNet/VAE replacement paths, and selective pointwise/spatial policies. `export-quantized --pointwise-convolution-policy feedforward` has been run for `ft_places2-q4`, producing `unet_quantized_pointwise_feedforward.safetensors` and a staged deployment folder of about `454 MiB`.

The experimental quantized VAE files reduce the measured deployment payload to 322.50 MB decimal / 307.38 MiB audited, but increase two-step image RMSE versus f16 from 0.000286 for default q4 to about 0.0115. The all-pointwise + spatial + VAE artifact reaches 200.50 MB decimal / 191.00 MiB audited, but its latent metrics are much worse. The benchmark suite now includes deterministic synthetic samples and a file-backed real-image smoke path; the next quantization work is broader real-image dataset coverage and a mixed policy between feed-forward-only and all-pointwise that trades size less aggressively.

## Known Limitations

- Quantized variants materialize compact quantized UNet parameters. By default only supported UNet `Linear` and `Embedding` modules are quantized; `ft_places2-q4` has additionally been re-exported with feed-forward 1x1 UNet convolutions packed as a size/fidelity experiment.
- The default/recommended q4 path keeps VAE weights f16 for stability. Experimental candidate views can use quantized VAE files, but current image metrics show a much larger RGB error.
- The repo includes a generated ONNX-backed benchmark report under `Docs/BenchmarkReports`, and the external fixture/report cache is under `/Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX-benchmarks`. Regenerating those references on a fresh machine still requires the Python reference dependencies, including `onnxruntime` and/or `diffusers`.
- The demo can import a source image from Files and supports direct paint/erase/clear mask editing over the source image. It does not yet import an external mask image file. Imported images start with an empty editable mask; the built-in sample starts with a circular mask for quick testing.
- Catalyst demo generation is verified through the autorun controller path. iPad Simulator UI launch is verified, but MLX Swift aborts while initializing its Metal stream/device on iOS Simulator, so the library and demo now reject simulator model execution before touching MLX.
- Physical iPad build, install, and model-container copy have been verified. The real-device autorun launch still needs an available/unlocked iPad; the latest 2026-06-24 `devicectl list devices` check reported the iPad as `unavailable`. The autorun status path now records timing and resident memory fields needed for that final device validation.
