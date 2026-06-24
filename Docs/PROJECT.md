# Moebius-MLX Project Writeup

## Goal

Port Moebius image inpainting to Swift/MLX so it can run locally on Apple Silicon macOS and in an iPadOS app. Mac Catalyst is the practical local demo target; iPadOS remains the primary deployment target.

Repository: `https://github.com/kylehowells/Moebius-MLX`

The GitHub workflow in `.github/workflows/ci.yml` covers source-only checks that can run without the external model artifact directory: Swift package tests, CLI help, shell/Python helper syntax, and checked-in benchmark report validation. Full model-artifact and Catalyst verification remains a local gate through `Scripts/verify_local.sh`.

The requested deliverables are:

- converted MLX model artifacts in f32, f16, and lower precision variants
- Swift MLX library and CLI
- verification against known-good behavior
- UIKit demo app that lets you compare model families and quant variants
- final writeup explaining the process, decisions, API, usage, and limits

## Source Material

- PyTorch/reference code: `/Users/kylehowells/Developer/ML-Models/Moebius/Moebius-Code`
- original checkpoints: `/Users/kylehowells/Developer/ML-Models/Moebius/Moebius-Models`
- ONNX/browser reference: `/Users/kylehowells/Developer/ML-Models/Moebius/moebius-web`
- converted MLX artifacts: `/Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX`

## Artifact Layout

The conversion script writes one directory per model family and precision:

```text
Moebius-MLX/
  ft_places2-f32/
    manifest.json
    unet.safetensors
    vae_encoder.safetensors
    vae_decoder.safetensors
  ft_places2-f16/
  ft_places2-q8/
  ...
```

There are 24 converted directories:

- families: `pretrained`, `ft_places2`, `ft_celebahq`, `ft_ffhq`
- precisions: `f32`, `f16`, `q8`, `q4`, `mxfp8`, `mxfp4`

Each `manifest.json` records the source checkpoint, image/latent sizes, scaling factor, scheduler config, output files, and precision metadata.

For quantized variants, the manifest also records:

```json
"unetQuantized": "unet_quantized.safetensors"
```

The loader prefers that materialized quantized UNet file when present.

## Runtime Shape Contract

Moebius uses fixed image and latent sizes:

- image: `512x512`
- latent: `64x64`
- latent channels: `4`

The UNet receives a 9-channel NHWC latent tensor:

1. noisy latents: 4 channels
2. binary latent mask: 1 channel
3. masked-image latents: 4 channels

Classifier-free guidance duplicates the batch:

- unconditional embedding IDs: `10..<20`
- conditional embedding IDs: `0..<10`

The VAE scaling factor is `0.13025`. This is deliberately not Stable Diffusion's common `0.18215`.

## Swift Implementation

The library is split into focused files:

- `MoebiusConfig.swift`: manifest, precision, scheduler, generation option types
- `MoebiusModel.swift`: artifact resolution, safetensors load, dtype conversion, quantization setup
- `MoebiusWeightMapping.swift`: PyTorch-to-MLX parameter name remapping
- `MoebiusUNet.swift`: Moebius UNet, lambda attention, depthwise separable blocks
- `MoebiusVAE.swift`: VAE encoder/decoder
- `MoebiusScheduler.swift`: DDIM scheduler and step math
- `MoebiusPipeline.swift`: latent denoising and CFG
- `MoebiusImagePipeline.swift`: image/mask preparation, VAE encode/decode, image output conversion

The main API is:

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

`imageFloats` are NHWC RGB values in `[-1, 1]`. `maskFloats` are NHWC single-channel values where `1` means inpaint this region. The returned array is NHWC RGB in `[0, 1]`.

See `Docs/API.md` for the command-level CLI reference and focused Swift API notes.

`MoebiusModelResolver` is intentionally usable from both command-line tools and apps. It searches explicit `preferredDirectory` / `modelRoot` arguments, `MOEBIUS_MLX_MODEL_DIR`, `MOEBIUS_MODEL_ROOT` as a demo/autorun compatibility alias, app bundle resource folders, app Application Support/Documents folders, and finally the local development path:

```text
/Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX
```

For iPad packaging, the expected model layout is one variant directory per model, for example `Moebius-MLX/ft_places2-f16/manifest.json`, placed in the app bundle, copied into Application Support, or imported into Documents. The resolver also accepts `Documents/<variant>/manifest.json`, which is useful because `devicectl device copy to --destination Documents` can copy the staged variant directories directly under Documents. The demo app enables `UIFileSharingEnabled` and `LSSupportsOpeningDocumentsInPlace`, so a physical iPad can receive staged model folders through Finder file sharing.

`Scripts/stage_models_for_ios.sh` now stages runtime packages rather than raw source directories. For quantized variants it copies only:

```text
manifest.json
unet_quantized.safetensors
vae_encoder.safetensors
vae_decoder.safetensors
```

For experimental VAE-quantized manifests, staging copies `vae_encoder_quantized.safetensors` and `vae_decoder_quantized.safetensors` instead of the f16 VAE files. It omits the original `unet.safetensors` because `MoebiusModel.load` uses `files.unetQuantized` when present. Use `--include-source-unet` only when intentionally preserving the source UNet for debugging. The current trimmed staged `ft_places2-q4` package uses the feed-forward pointwise q4 artifact, is about `454 MiB`, and has been loaded directly through the CLI.

The staging helper creates the expected folder layout:

```bash
Scripts/stage_models_for_ios.sh \
  --output /tmp/moebius-ipad-models \
  ft_places2-f16 ft_places2-q4
```

Then copy `/tmp/moebius-ipad-models/Moebius-MLX` into the installed app's Documents container. The resolver will find:

```text
Documents/Moebius-MLX/ft_places2-f16/manifest.json
Documents/Moebius-MLX/ft_places2-q4/manifest.json
Documents/ft_places2-f16/manifest.json
Documents/ft_places2-q4/manifest.json
```

Lower-level callers can use:

- `MoebiusRuntime.supportsModelExecution`
- `MoebiusPipeline.predictNoiseCFG(...)`
- `MoebiusPipeline.runLatentFixture(...)`
- `MoebiusPipeline.runImage(...)`

## Quantization Decisions

MLX supports affine quantization and MXFP modes for supported modules. The converted variants are:

- `f32`: f32 weights
- `f16`: f16 weights
- `q8`: materialized MLX affine 8-bit quantized UNet parameters
- `q4`: materialized MLX affine 4-bit quantized UNet parameters
- `mxfp8`: materialized MLX MXFP8 quantized UNet parameters
- `mxfp4`: materialized MLX MXFP4 quantized UNet parameters

The standard converted q/mxfp variants quantize supported UNet `Linear` and `Embedding` modules. Modules whose dimensions do not fit the requested group size are skipped. The default deployment recommendation keeps VAE weights unquantized because the VAE is visually sensitive and the measured VAE-quantized candidates have a much larger RGB error.

The q/mxfp conversion directories keep the f16 `unet.safetensors` as source/backing data and also include `unet_quantized.safetensors`. Current quantized UNet files are around `402-413 MiB` versus `432 MiB` for the f16 UNet because convolutional weights remain f16. Runtime staging excludes that source UNet, so deployment size is `unet_quantized + VAE encoder + VAE decoder`, not the full conversion directory size.

The repo also includes experimental `ft_places2` candidate views that opt into additional packed convolution/VAE files:

- `ft_places2-q4-ff-spatial-vae`: feed-forward pointwise and spatial sampler UNet packing plus quantized VAE files, about 322.50 MB audited.
- `ft_places2-q4-all-spatial-vae`: broader pointwise/spatial UNet packing plus quantized VAE files, about 200.28 MB audited / 191 MiB staged.

Those candidates are useful for measuring the size target, but current image and latent benchmarks do not support making either one the default quality recommendation.

## Demo App

The demo app lives at:

```text
Demo/MoebiusDemo
```

It is a UIKit app with:

- `AppDelegate`
- programmatic `SceneDelegate`
- no main storyboard
- a Launch Screen storyboard only
- segmented controls for model family and precision
- a stepper for 2-24 denoising steps
- synthetic sample image, imported-image support, mask preview, and output preview
- Load image and Reset sample controls
- a run-history comparison panel showing recent variants, status, step count, runtime, output mean, and manifest-selected deployment MB
- a run button that calls the same `MoebiusPipeline.runImageRGBFloats` API as the CLI image smoke path
- a Compare quants button that runs f32, f16, q8, q4, mxfp8, and mxfp4 for the selected model family on the same sample
- additional `ft_places2` precision options for the experimental `q4 322M` and `q4 200M` candidates, so that family compares eight visible variants
- comparison mode records per-variant errors and continues, so an iPad staged with only a subset of variants can still compare the available models
- paste-back enabled so unmasked pixels come from the input image
- an environment-driven autorun path for repeatable Catalyst verification:
  - `MOEBIUS_DEMO_AUTORUN=1`
  - `MOEBIUS_DEMO_COMPARE=1` for the visible-variant comparison path
  - `MOEBIUS_DEMO_MODEL=ft_places2`
  - `MOEBIUS_DEMO_PRECISION=f16` or a candidate variant name such as `ft_places2-q4-all-spatial-vae`
  - `MOEBIUS_DEMO_STEPS=2`
  - `MOEBIUS_DEMO_AUTORUN_STATUS=/tmp/moebius-demo-autorun-status.txt`

If `MOEBIUS_DEMO_AUTORUN=1` is set and `MOEBIUS_DEMO_AUTORUN_STATUS` is not set, the app writes the latest autorun status to `Documents/moebius-demo-autorun-status.txt` and also prints the status. This makes physical-device runs observable through `devicectl` app-container file copy.

The app builds for both Mac Catalyst and iOS Simulator. Catalyst is currently the easiest local test target.

Imported images are loaded through a document picker, center-cropped/resized to the 512x512 tensor shape, normalized to `[-1, 1]`, and paired with the same fixed circular inpaint mask used by the synthetic sample. Mask drawing/import is intentionally not part of this pass yet.

## Verification

Commands run successfully:

```bash
swift build
swift test
swift run moebius-mlx verify-fixture ft_places2-f32
swift run moebius-mlx image-smoke ft_places2-f16 --steps 2
swift run moebius-mlx image-smoke ft_places2-q4 --steps 2
swift run moebius-mlx compare-image ft_places2 --steps 2
Scripts/verify_local.sh --demo-builds
Scripts/stage_models_for_ios.sh --output /tmp/moebius-ios-stage-test ft_places2-q4
```

`Scripts/verify_local.sh` is the repeatable local verification bundle. By default it checks:

- all 24 expected model artifact directories
- the two stable experimental `ft_places2` q4 candidate views
- all required manifest, UNet, VAE encoder, and VAE decoder files
- all 16 quantized UNet files and their manifest references
- helper script syntax, including benchmark, staging, and device autorun scripts
- `swift test`, including scheduler metadata, runtime-support metadata, resolver search paths, and missing-model error payloads
- representative `inspect` loads for f32, f16, and q4
- `verify-fixture ft_places2-f32`
- `image-smoke ft_places2-f16 --steps 2`
- `image-smoke ft_places2-q4 --steps 2`
- `compare-image ft_places2 --steps 2`
- `benchmark-image ft_places2-q4 --reference-model ft_places2-f16 --steps 2 --samples gradient-circle`
- trimmed iPad staging and load checks for `ft_places2-q4` and `ft_places2-q4-all-spatial-vae`

Optional flags:

```bash
Scripts/verify_local.sh --all-inspects --demo-builds
```

`--all-inspects` loads all 24 variants through the Swift CLI. `--demo-builds` also runs the Catalyst and generic iOS Simulator Xcode builds.

Latest local verification on 2026-06-24:

- `Scripts/verify_local.sh --all-inspects` passed.
- All 24 standard converted variants loaded through `moebius-mlx inspect`: `pretrained`, `ft_places2`, `ft_celebahq`, and `ft_ffhq` across `f32`, `f16`, `q8`, `q4`, `mxfp8`, and `mxfp4`.
- The same run passed Swift tests, f32 fixture smoke, f16/q4 image smoke, `compare-image ft_places2 --steps 2`, the f16-referenced q4 image benchmark, and trimmed staging/load checks for `ft_places2-q4` and `ft_places2-q4-all-spatial-vae`.
- `Scripts/verify_local.sh --demo-builds` also passed on 2026-06-24, covering the Catalyst build, Catalyst autorun comparison paths, and generic iOS Simulator build.

The iPad staging helper was smoke-tested with one quantized variant and produced:

```text
/tmp/moebius-ios-stage-test/Moebius-MLX/ft_places2-q4/manifest.json
```

The current `ft_places2-q4` manifest points at the feed-forward pointwise q4 artifact. Staging that variant produces a trimmed folder of about `454 MiB`, omits `unet.safetensors`, and reloads through `moebius-mlx inspect`.

The 200 MB candidate also stages cleanly:

```bash
Scripts/stage_models_for_ios.sh --output /tmp/moebius-ios-candidate-stage ft_places2-q4-all-spatial-vae
```

Result:

```text
staged ft_places2-q4-all-spatial-vae (191M)
```

Physical iPad validation progressed to build, install, and model provisioning:

```text
Device: Kyle‘s iPad Pro (18.6)
xcodebuild destination UDID: 00008103-001460290E31001E
devicectl CoreDevice id: 4B1F3A7A-CB7A-5D9A-85FF-D7578C7EC4C1
```

The physical build passed:

```bash
xcodebuild -project Demo/MoebiusDemo/MoebiusDemo.xcodeproj \
  -scheme MoebiusDemo \
  -destination 'platform=iOS,id=00008103-001460290E31001E' \
  build
```

The app installed through `devicectl`:

```bash
xcrun devicectl device install app \
  --device 4B1F3A7A-CB7A-5D9A-85FF-D7578C7EC4C1 \
  /Users/kylehowells/Library/Developer/Xcode/DerivedData/MoebiusDemo-cjsunmzjjdrzelbaktvwigzgnnwb/Build/Products/Debug-iphoneos/MoebiusDemo.app
```

One quantized model was staged and copied to the app data container:

```bash
Scripts/stage_models_for_ios.sh --output /tmp/moebius-ipad-stage ft_places2-q4
xcrun devicectl device copy to \
  --device 4B1F3A7A-CB7A-5D9A-85FF-D7578C7EC4C1 \
  --source /tmp/moebius-ipad-stage/Moebius-MLX \
  --destination Documents \
  --domain-type appDataContainer \
  --domain-identifier com.ikyle.MoebiusDemo \
  --remove-existing-content false
```

Readback confirmed the model manifest on the device:

```bash
xcrun devicectl device copy from \
  --device 4B1F3A7A-CB7A-5D9A-85FF-D7578C7EC4C1 \
  --source Documents/ft_places2-q4/manifest.json \
  --destination /tmp/moebius-ipad-readback/manifest.json \
  --domain-type appDataContainer \
  --domain-identifier com.ikyle.MoebiusDemo
```

The copied manifest reported `modelName: ft_places2` and `precision: q4`.

The remaining physical-device runtime command is:

```bash
Scripts/run_device_autorun.sh --variant ft_places2-q4
```

The helper wraps the known-good manual sequence: device build, app install, model staging, model copy into the app Documents container, autorun launch, and status readback. It can also run candidate variants:

```bash
Scripts/run_device_autorun.sh --variant ft_places2-q4-all-spatial-vae
Scripts/run_device_autorun.sh --model ft_places2 --compare
Scripts/run_device_autorun.sh --model ft_places2 --compare --dry-run
Scripts/run_device_autorun.sh --model ft_places2 --compare --wait-seconds 300
```

`--compare` stages every visible precision for the selected model. For `ft_places2`, it stages the six standard precision variants plus the `q4 322M` and `q4 200M` package-size candidates, matching the demo comparison UI. `--dry-run` prints that resolved staging list and the autorun launch environment before touching CoreDevice, which keeps the next physical-device attempt auditable while the iPad is unavailable. `--wait-seconds` can be used for the actual attempt to poll for an unlocked/connected device before build, install, copy, and launch.

The latest successful provisioning attempt copied the original staged q4 package to the iPad app container and read the manifest back. A later 2026-06-24 continuation check could not launch the app because CoreDevice reported the iPad as unavailable:

```text
Kyle‘s iPad Pro  4B1F3A7A-CB7A-5D9A-85FF-D7578C7EC4C1  unavailable
```

The device helper was tested against the current unavailable CoreDevice state with:

```bash
Scripts/run_device_autorun.sh --variant ft_places2-q4-all-spatial-vae --skip-build --skip-install --skip-copy
```

It correctly exited before build/install with:

```text
Device is not available:
Kyle‘s iPad Pro ... unavailable ...
```

The current feed-forward q4 package for the next physical-device attempt is staged at `/tmp/moebius-ipad-feedforward-stage/Moebius-MLX/ft_places2-q4`.

All 24 converted variants load through:

```bash
for model in pretrained ft_places2 ft_celebahq ft_ffhq; do
  for precision in f32 f16 q8 q4 mxfp8 mxfp4; do
    swift run moebius-mlx inspect "$model-$precision"
  done
done
```

All 16 quantized variants have materialized quantized UNet artifacts:

```bash
find /Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX \
  -maxdepth 2 -name unet_quantized.safetensors
```

Catalyst build:

```bash
xcodebuild -project Demo/MoebiusDemo/MoebiusDemo.xcodeproj \
  -scheme MoebiusDemo \
  -destination 'platform=macOS,variant=Mac Catalyst' \
  build
```

iOS Simulator build:

```bash
xcodebuild -project Demo/MoebiusDemo/MoebiusDemo.xcodeproj \
  -scheme MoebiusDemo \
  -destination 'generic/platform=iOS Simulator' \
  build
```

The generic iOS Simulator build was re-run after adding the `ft_places2` candidate precision options and passed from a fresh derived-data folder:

```text
** BUILD SUCCEEDED ** [79.440 sec]
```

The Catalyst app was launched with autorun enabled through the real `ViewController` path:

```text
done ft_places2-f16 steps=2 model_mb=619.89 min=0.0 max=1.0 mean=0.50244683
```

The Catalyst app was also launched through the same autorun path using the current feed-forward pointwise `ft_places2-q4` artifact:

```text
done ft_places2-q4 steps=2 model_mb=475.67 min=0.0 max=1.0 mean=0.5024325
```

The Catalyst comparison autorun path was also launched through the real `ViewController` path. It now covers the six standard precision variants plus the two visible `ft_places2` q4 package-size candidates:

```text
done comparison 8/8 variants errors=0 model_mbs=ft_places2-f32:1239.58,ft_places2-f16:619.89,ft_places2-q8:600.23,ft_places2-q4:475.67,ft_places2-mxfp8:599.56,ft_places2-mxfp4:589.09,ft_places2-q4-ff-spatial-vae:322.50,ft_places2-q4-all-spatial-vae:200.50
```

The current candidate selector path was then launched through the same Catalyst autorun mechanism using `MOEBIUS_DEMO_PRECISION=ft_places2-q4-all-spatial-vae`:

```text
done ft_places2-q4-all-spatial-vae steps=2 model_mb=200.50 min=0.0 max=1.0 mean=0.5017574
```

The iPad Simulator app was also built and launched with the corrected bundle id, `com.ikyle.MoebiusDemo`. The runtime UI snapshot showed the programmatic `SceneDelegate` root controller with:

- model tabs: `ft_places2`, `ft_celebahq`, `ft_ffhq`, `pretrained`
- precision tabs: `f32`, `f16`, `q8`, `q4`, `mxfp8`, `mxfp4`
- `ft_places2` also adds `q4 322M` and `q4 200M` candidate precision tabs
- `Steps: 8`
- `Load image`
- `Reset sample`
- `Run sample`
- `Compare quants`
- `Runs` with six comparison rows
- `Sample`, `Mask`, and `Result` labels

The simulator run path also updates the run-history panel. On iOS Simulator the single-run first row becomes:

```text
ft_places2-f16
Unsupported on simulator 8 steps 0.0s
```

The simulator comparison path fills all six rows without touching MLX:

```text
ft_places2-f32
ft_places2-f16
ft_places2-q8
ft_places2-q4
ft_places2-mxfp8
ft_places2-mxfp4
```

During investigation, iPad Simulator autorun entered the controller path and then MLX Swift aborted while initializing its default stream/device before reading safetensors:

```text
/Applications/Xcode.app/.../usr/include/c++/v1/string:1093:
assertion __s != nullptr failed: basic_string(const char*) detected nullptr
```

The crash report pointed at `MLX.loadArrays(url:)`'s default `StreamOrDevice.cpu` argument, `mlx_default_cpu_stream_new`, and `mlx::core::metal::Device::Device()`. The package now exposes `MoebiusRuntime.supportsModelExecution` and rejects model execution on iOS Simulator before touching MLX. Current simulator autorun result:

```text
error ft_places2-f16 iOS Simulator cannot run MLX models
```

Recorded f32 UNet zero-fixture parity against the ONNX reference:

```text
ONNX min/max/mean: -0.3489277363 / 0.1178752705 / -0.0739606619
MLX  min/max/mean: -0.3489233255 / 0.1178750768 / -0.0739595667
max_abs:  2.0820647478103638e-05
mean_abs: 1.8807565993483877e-06
p95_abs:  4.947185516357422e-06
p99_abs:  8.098501712083817e-06
```

Lower precision zero-fixture drift recorded against ONNX:

```text
f16:   max_abs 0.1402668, mean_abs 0.0607061
q8:    max_abs 0.1432423, mean_abs 0.0606215
q4:    max_abs 0.1487513, mean_abs 0.0569222
mxfp8: max_abs 0.1530783, mean_abs 0.0505010
mxfp4: max_abs 0.1726358, mean_abs 0.0505319
```

Interpretation: f32 is the strict parity target. Lower precision variants load and run, but quality needs visual/runtime validation rather than treating the zero synthetic fixture as a pass/fail parity target.

The CLI also includes `compare-image`, which runs the same synthetic image/mask sample across precision variants and compares `[0, 1]` RGB outputs against the first listed precision. Recorded `ft_places2` output with f16 as the reference:

```text
variant          mean        max_abs_vs_ref  mean_abs_vs_ref
ft_places2-f16   0.50244707  -               -
ft_places2-q8    0.5024465   0.00094777346   0.000006535905
ft_places2-q4    0.50244474  0.0038945675    0.000024847031
ft_places2-mxfp8 0.5024485   0.0074070096    0.000042056134
ft_places2-mxfp4 0.5024302   0.010975122     0.00005909382
```

That smoke comparison is useful, but it is not enough to prove the port is faithful. The stronger benchmark path now uses a deterministic latent fixture generated by `Scripts/generate_reference_fixture.py` and consumed by:

```bash
swift run moebius-mlx benchmark-fixture ft_places2-f32 \
  --fixture /Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX-benchmarks/fixtures/ft_places2-onnx-steps20 \
  --reference-name onnx
```

The fixture pins:

- source image and binary mask
- initial latent noise, including noise offset
- scheduler timesteps
- guidance scale
- strength
- denoising step count
- final latent tensor from ONNX or PyTorch

The benchmark reports:

- `max_abs`: worst single-value difference, good at catching layout/channel/order bugs
- `mean_abs`: average absolute drift
- `rmse`: squared-error drift, more sensitive to larger differences
- `p99_abs`: tail error without letting one value dominate the whole read
- `psnr_db`: image-style signal/noise summary for easier run-to-run comparison
- `cosine_similarity`: latent-direction agreement
- total model MB, UNet bytes, and VAE bytes
- `load_seconds`: model load/setup cost
- `inference_seconds`: fixed latent-denoising runtime, the main quant speed comparison
- `total_seconds`: cold model load plus inference

Graph/report generation is handled by:

```bash
Scripts/run_benchmark_report.sh \
  --family ft_places2 \
  --reference onnx \
  --fixture /Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX-benchmarks/fixtures/ft_places2-onnx-steps20 \
  --output-dir /Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX-benchmarks/reports/ft_places2-onnx-steps20
```

It writes:

- `benchmark.tsv`
- `summary.md`
- `fidelity_errors.png`
- `performance_timings.png`
- `model_size.png`
- `speed_fidelity_tradeoff.png`
- `size_fidelity_tradeoff.png`

The intended acceptance ladder is:

- f32 MLX vs ONNX/PyTorch: strict parity target, expected to be floating-point/layout-level error only
- f16 MLX vs ONNX/PyTorch: small numerical drift, but still very close
- q8/q4/mxfp8/mxfp4 vs the same reference: measured quality/size tradeoff, not automatically acceptable just because it runs

For the size target, the benchmark prints real artifact size from the loaded model directory. `moebius-mlx inspect` now prints the same manifest-selected deployment files plus loaded dtype payloads, so package-size questions can be answered from the Swift toolchain:

```text
swift run moebius-mlx inspect ft_places2-q4
deployment mb: 475.67
loaded tensor payload by dtype:
  float16: approx=425.34 MB
  uint32: approx=50.16 MB

swift run moebius-mlx inspect ft_places2-q4-all-spatial-vae
deployment mb: 200.50
loaded tensor payload by dtype:
  uint32: approx=145.89 MB
  float16: approx=54.39 MB
```

This matters because the default q4 package still includes a large fp16 VAE and only quantizes selected UNet modules. Experimental VAE + broader UNet convolution packing now reaches about 200 MB, but the current 200 MB candidate has substantial latent drift. The benchmark is the gate for whether those reductions are acceptable.

Current dependency state: the default Python environment has `torch`, `numpy`, `PIL`, and `safetensors`, but is missing `onnxruntime` and `diffusers`. The checked-in report was created from an ONNX-backed fixture already present under the model workspace; regenerating that fixture from scratch requires `onnxruntime`, and regenerating PyTorch references requires `diffusers`. The Swift `benchmark-fixture` command has been smoke-tested with a synthetic fixture and used for the checked-in ONNX-backed graph report.

The full graph/report path has also been smoke-tested with a synthetic fixture:

```bash
Scripts/run_benchmark_report.sh \
  --fixture /tmp/moebius-empty-fixture \
  --reference synthetic-zero \
  --steps 2 \
  --output-dir /tmp/moebius-benchmark-report-smoke
```

That produced all five PNG charts and a markdown table. These numbers are intentionally not fidelity evidence because the reference tensor is synthetic zeros; the run only validates that benchmark collection, timing, plotting, and report generation work end to end.

A real ONNX-backed 20-step `ft_places2` fixture and graph report were generated for the demo-visible variants at:

```text
/Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX-benchmarks/fixtures/ft_places2-onnx-steps20
Docs/BenchmarkReports/ft_places2-onnx-steps20-visible-variants
```

Recorded metrics:

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

The checked-in graph report contains:

- `summary.md`
- `fidelity_errors.png`
- `performance_timings.png`
- `model_size.png`
- `speed_fidelity_tradeoff.png`
- `size_fidelity_tradeoff.png`

Interpretation: ONNX is close to PyTorch over this latent fixture, and f32 MLX is very close to ONNX. Current f16 and quantized variants are smaller and sometimes faster, but they accumulate substantial 20-step latent drift. They are useful for tradeoff exploration, not yet proven quality-equivalent deployment artifacts. The current `ft_places2-q4` manifest points at the feed-forward pointwise policy artifact, reducing the runtime payload to 475.67 MB decimal / about 454 MiB staged. The `q4 322M` candidate adds spatial and VAE quantized files and is the fastest row in the current run, but with slightly worse latent drift. The `q4 200M` candidate reaches the requested package-size target while increasing 20-step latent drift substantially, so the package-size target remains unmet for a fidelity-acceptable deployment artifact.

An additional experimental artifact, `unet_quantized_pointwise_feedforward_spatial.safetensors`, packs feed-forward 1x1 convolutions plus eligible spatial down/up-sampler 3x3 convolutions through the im2col + `quantizedMM` path. It is not the default manifest target because it saves size but slightly worsens the focused ONNX fixture:

```text
variant                    model MB  RMSE      cosine      inference s
ft_places2-q4-ff-pw        475.67    0.176075  0.97774184  14.363
ft_places2-q4-ff-spatial   442.56    0.178230  0.97694558  15.436
```

The focused q4 spatial graph report is in:

```text
/Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX-benchmarks/reports/ft_places2-onnx-steps20-q4-spatial
```

The next experimental candidate also adds `vae_encoder_quantized.safetensors` and `vae_decoder_quantized.safetensors`, packing eligible VAE `Linear` and Conv2d weights. The latent fixture does not exercise VAE reconstruction/decode quality, so the RMSE/cosine values remain identical to the UNet-only spatial candidate. The useful evidence in this report is package size, load time, denoising runtime, plus a separate two-step `image-smoke` run that loaded the quantized VAE path and produced bounded `[0, 1]` output.

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

The audited deployment payload is 200.28 MB decimal / 191.00 MiB. It loads and completes `image-smoke --steps 2`, but this should be considered a boundary experiment because both latent and image-level metrics show meaningful quality loss versus larger candidates.

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

The image benchmark path now also supports deterministic multi-sample aggregation:

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

The `all` sample set covers four deterministic image/mask profiles: gradient/circle, checker/square, radial/band, and diagonal/stripes. Aggregate two-step metrics:

```text
variant                         model MB  image RMSE  p99 abs   cosine      inference s
ft_places2-q4-default           475.67    0.000343    0.001569  0.99999982  5.038
ft_places2-q4-ff-spatial-vae    322.50    0.014153    0.068663  0.99971294  15.224
ft_places2-q4-all-spatial-vae   200.50    0.014082    0.068348  0.99971575  6.968
```

This broader image check preserves the earlier conclusion: default q4 remains the best quality/speed tradeoff; the 200 MB candidate is a real package-size target but carries a much larger RGB error. The all-samples graph report is in:

```text
/Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX-benchmarks/reports/ft_places2-image-f16-steps2-q4-candidates-all-samples
```

The same report path supports file-backed image/mask benchmarks:

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

The CLI center-crops/resizes the image to 512x512 and thresholds white mask pixels as the inpaint region. A generated PNG/mask smoke check produced:

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

`Scripts/audit_model_size.py` gives a tensor-level size breakdown. For the current trimmed `ft_places2-q4` feed-forward pointwise deployment:

```text
total:                  453.47 MiB
UNet quantized file:    293.91 MiB
VAE decoder:             94.40 MiB
VAE encoder:             65.16 MiB
UNet 1x1 conv/proj:     165.58 MiB
UNet 3x3 conv/proj:      45.40 MiB
VAE conv weights:       155.48 MiB
actual uint32 packed:    47.83 MiB
ideal q4 no metadata:   107.39 MiB
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

The current q4/mxfp4 variants are therefore not full-model q4 packages. The original quantized artifacts pack only the MLX-quantized `Linear`/`Embedding` modules; Conv2d-heavy UNet projections, depthwise/pointwise feed-forward layers, relative-position tensors, and the VAE remain mostly f16. The same caveat explains why the current q8/mxfp8 packages are still near the f16 package size: `ft_places2-q8` deploys at about 572 MiB because it is a mostly-f16 pipeline bundle with a partially quantized UNet and f16 VAE files, not a raw one-byte-per-parameter fp8 export. Inspection of MLX Swift shows quantized 2D matrix packing/matmul plus `QuantizedLinear`/`QuantizedEmbedding`, but no built-in `QuantizedConv2d`. The package now includes `MoebiusQuantizedPointwiseConv2d`, a packed NHWC convolution backed by im2col + `quantizedMM`; the Swift test suite checks q8 output against exact float references for both 1x1 and 3x3 kernels, the explicit opt-in UNet/VAE replacement paths, and selective pointwise/spatial policies. `export-quantized --pointwise-convolution-policy feedforward` has been run for `ft_places2-q4`, producing `unet_quantized_pointwise_feedforward.safetensors` and reducing staged deployment size to about `454 MiB`. The experimental spatial-sampler artifact reduces this to about `422 MiB` staged, at a small fidelity/speed cost. Adding experimental quantized VAE files reduces the audited deployment payload to about `307 MiB`, but increases two-step image RMSE versus f16 from 0.000286 for default q4 to about 0.0115. The all-pointwise + spatial + VAE artifact reaches 200.50 MB decimal / 191.00 MiB audited, but its latent metrics are much worse. The benchmark suite now includes deterministic synthetic samples and a file-backed real-image smoke path; next quantization work is broader real-image dataset coverage and a mixed UNet policy between feed-forward-only and all-pointwise that trades size less aggressively.

## Limitations And Next Work

- The repo includes a generated ONNX-backed benchmark report under `Docs/BenchmarkReports`, and the external fixture/report cache is under `/Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX-benchmarks`. Regenerating those references on a fresh machine still requires the Python reference dependencies, including `onnxruntime` and/or `diffusers`.
- Quantized variants materialize compact quantized UNet parameters. By default only supported `Linear` and `Embedding` modules are quantized. `ft_places2-q4` has additionally been re-exported with feed-forward 1x1 UNet convolutions packed, and experimental non-default artifacts also pack eligible spatial sampler 3x3 convolutions and VAE conv/linear weights; remaining 1x1 attention/resnet, depthwise, and relative-position weights are still mostly f16.
- Catalyst demo generation has been verified through the actual view-controller autorun path. iPad Simulator UI launch is verified, but MLX Swift currently aborts during Metal stream/device initialization on iOS Simulator. The library and demo now fail gracefully there.
- Physical iPad build, install, and model-container copy are verified. Real-device model execution and timing/memory validation still need one available unlocked-device autorun/manual run; the latest 2026-06-24 `devicectl list devices` check reported the iPad as unavailable.
- The demo can import source images from Files, but mask editing/import is not implemented. It currently uses a fixed circular mask for both synthetic and imported samples.
