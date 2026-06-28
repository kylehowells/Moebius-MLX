# Moebius-MLX Project Notes

Moebius-MLX is a Swift and MLX port of Moebius image inpainting for Apple Silicon macOS, Mac Catalyst, and iPadOS. The repository contains:

- a SwiftPM library product, `MoebiusMLX`
- a command-line tool, `moebius-mlx`
- conversion, staging, and benchmark scripts
- a UIKit demo app for Mac Catalyst and iPadOS
- checked-in benchmark reports for reproducible comparison

The model artifacts themselves are intentionally not committed to this source repository. Published variants are distributed as one Hugging Face model repo per family/precision, and the library can also load bundled, app-managed, or user-supplied model directories.

## Source Material

The port is based on the original PyTorch Moebius checkpoints and architecture. The ONNX/WebGPU export is used as an additional reference runtime for benchmarking, not as the source for MLX conversion.

Portable script arguments are used instead of machine-specific paths:

- `--pytorch-code-root`
- `--pytorch-weights-root`
- `--vae-dir`
- `--onnx-root`
- `--mlx-model-root`

If those paths are omitted, the public benchmark script can download the required public artifacts into `BenchmarkArtifacts/`, which is ignored by Git.

## Model Families

The converted model set has four families:

- `pretrained`: base Moebius checkpoint.
- `ft_places2`: checkpoint fine-tuned for Places2 inpainting.
- `ft_celebahq`: checkpoint fine-tuned for CelebA-HQ face inpainting.
- `ft_ffhq`: checkpoint fine-tuned for FFHQ face inpainting.

Each family has standard precision variants:

- `f32`
- `f16`
- `q8`
- `q4`
- `mxfp8`
- `mxfp4`

`ft_places2` also includes two experimental q4 size-target variants:

- `ft_places2-q4-ff-spatial-vae`: about 322 MB.
- `ft_places2-q4-all-spatial-vae`: about 200 MB.

Those two variants are useful for size and quality experiments. They are not presented as drop-in replacements for the default q4 package because the benchmark reports show a larger fidelity loss.

## Artifact Layout

A model variant directory contains a manifest and the active weight files:

```text
ft_places2-q4/
  manifest.json
  unet.safetensors
  unet_quantized.safetensors
  vae_encoder.safetensors
  vae_decoder.safetensors
```

Candidate variants may select more specialized files:

```text
unet_quantized_pointwise_feedforward.safetensors
unet_quantized_pointwise_feedforward_spatial.safetensors
unet_quantized_pointwise_all_spatial.safetensors
vae_encoder_quantized.safetensors
vae_decoder_quantized.safetensors
```

The manifest decides which files are active at runtime. `MoebiusModelDeploymentInfo` reports the manifest-selected deployment size, so size metrics match what a staged or downloaded package actually needs.

Moebius uses fixed tensor sizes:

- input/output image: `512x512`
- latent: `64x64`
- latent channels: `4`
- UNet input channels: `9` (`noisy latent + mask + masked-image latent`)
- VAE scaling factor: `0.13025`

This is a released-checkpoint constraint, not just a convenience added by the MLX port. The original PyTorch wrapper can resize input images before inference, but the shipped model config builds the UNet for a `512x512` image and therefore a `64x64` latent. The lambda cross-attention path has learned relative-position tensors tied to the square latent grids used by the trained model (`64x64`, `32x32`, and `16x16`). Non-square latent grids and other native resolutions are outside the supported behavior of these weights.

App or CLI front ends can still accept arbitrary photos by mapping them into the square model workspace with crop, letterbox, or tile preprocessing. That preprocessing must transform the mask consistently and map the result back afterward. It should be documented as wrapper behavior, not as arbitrary-resolution model support.

## Swift Library Structure

The library is split into focused implementation files:

- `MoebiusConfig.swift`: manifest, precision, scheduler, and generation-option types.
- `MoebiusModel.swift`: model resolution, downloads, safetensors load, dtype conversion, quantization setup, and runtime support checks.
- `MoebiusModelDeploymentInfo.swift`: active deployment file and size accounting.
- `MoebiusWeightMapping.swift`: PyTorch-to-MLX parameter name remapping.
- `MoebiusUNet.swift`: Moebius UNet implementation.
- `MoebiusVAE.swift`: VAE encoder/decoder implementation.
- `MoebiusQuantizedConvolution.swift`: packed quantized convolution support used by experimental variants.
- `MoebiusScheduler.swift`: DDIM scheduler math.
- `MoebiusPipeline.swift`: latent denoising and classifier-free guidance.
- `MoebiusImagePipeline.swift`: image/mask preparation, VAE encode/decode, paste-back, and RGB output conversion.

Typical use:

```swift
import MoebiusMLX

let model = try await MoebiusModel.fromPretrained(
	"ft_places2-q4",
	onDownloadProgress: { progress in
		print(progress.fractionCompleted)
	}
)

let pipeline = MoebiusPipeline(model: model)
let input = MoebiusImageInput(imageNHWC: imageFloats, maskNHWC: maskFloats)
let options = MoebiusGenerationOptions(
	steps: 8,
	guidanceScale: 2,
	seed: 7,
	strength: 0.99,
	pasteBack: true
)

let outputRGB01 = try pipeline.runImageRGBFloats(
	input: input,
	options: options
)
```

`imageFloats` must be NHWC RGB normalized to `[-1, 1]`. `maskFloats` must be NHWC single-channel values where `1` marks the inpaint region. The returned image is NHWC RGB clamped to `[0, 1]`.

See [API.md](API.md) for the full API and CLI reference.

## Model Resolution

`MoebiusModel.load(from:)` loads an already available local model directory. Use it for bundled models, app-managed downloads, or user-selected model folders.

`MoebiusModel.fromPretrained(...)` resolves or downloads a model from:

- a local model directory
- a direct `manifest.json` path
- a known variant name such as `ft_places2-q4`
- a Hugging Face repo id
- a Hugging Face model URL

Known variants map to default Hugging Face repo ids such as:

```text
ft_places2-q4 -> iky1e/moebius-ft-places2-mlx-q4
ft_ffhq-f16 -> iky1e/moebius-ft-ffhq-mlx-f16
```

Download progress is exposed through `MoebiusDownloadProgress`, including byte counts, completion fraction, and optional transfer speed.

Default materialized snapshot locations:

- macOS with `HF_HUB_CACHE`: `$HF_HUB_CACHE/moebius-mlx`
- macOS with `HF_HOME`: `$HF_HOME/hub/moebius-mlx`
- macOS fallback: `~/.cache/huggingface/hub/moebius-mlx`
- iOS / sandboxed app fallback: app `Library/Application Support/Moebius-MLX/HuggingFace`

Apps can pass `downloadBase:` to choose a different materialized snapshot location, or bypass Hugging Face entirely by calling `MoebiusModel.load(from:)`.

## Quantization

The standard lower-precision packages quantize supported UNet `Linear` and `Embedding` modules. Convolution-heavy portions of the UNet and the VAE remain fp16 in the standard variants, which is why q8/q4 packages are smaller than f16 but not as small as one byte or half a byte per full model parameter.

The default q4 package keeps the VAE unquantized because VAE quantization is visually sensitive in the current benchmarks. The experimental `ft_places2` size-target variants additionally pack selected pointwise/spatial convolutions and quantized VAE files to reduce deployment size.

Current benchmark interpretation:

- `f32` is the strict parity target and should stay very close to the reference runtime.
- `f16`, `q8`, `q4`, `mxfp8`, and `mxfp4` are size/performance tradeoffs.
- The 322 MB and 200 MB q4 variants are useful experiments, but their fidelity loss is higher than the default q4 package.

Use the checked-in benchmark reports for measured numbers rather than treating precision names as quality guarantees.

## Benchmarks

The canonical public benchmark is [BenchmarkReports/ft_places2-cross-runtime-steps20](BenchmarkReports/ft_places2-cross-runtime-steps20/). It compares deterministic latent denoising across:

- PyTorch CPU and MPS
- ONNX Runtime providers available on the machine
- MLX f32/f16/quant variants

Run it with:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements-benchmarks.txt
./.venv/bin/python Scripts/run_cross_runtime_benchmark.py
```

The benchmark writes:

- `benchmark.csv`
- `README.md`
- `fidelity_errors.png`
- `performance_timings.png`
- `model_size.png`
- `speed_fidelity_tradeoff.png`
- `size_fidelity_tradeoff.png`
- `latent_contact_sheet.png`
- raw fixture and latent tensors

The CSV and table include all requested rows. Charts omit CPU baselines and very slow provider fallback rows by default so the relevant GPU/MLX timing range remains readable.

The older [BenchmarkReports/ft_places2-onnx-steps20-visible-variants](BenchmarkReports/ft_places2-onnx-steps20-visible-variants/) report is retained as supplemental history for demo-visible MLX variants against an ONNX latent fixture.

## Demo App

The demo app is in `Demo/MoebiusDemo`.

It is a storyboard-free UIKit app with:

- `AppDelegate`
- programmatic `SceneDelegate`
- Launch Screen storyboard only
- source image and mask editor
- generated result view
- side inspector for model, precision, steps, guidance, seed, paste-back, brush size, paint/erase, sample reset, clear mask, and quant comparison
- Photo Library and Files import
- Hugging Face download prompt when a selected model is missing locally
- autorun support for repeatable Catalyst and physical-device checks

On narrow devices, the inspector opens as a slide-over panel behind a dimming overlay. Imported images start with an empty editable mask. The built-in sample starts with a circular mask so the app can run immediately after a model is available.

See [../Demo/MoebiusDemo/README.md](../Demo/MoebiusDemo/README.md) for build and provisioning details.

## Verification

Source-only validation:

```bash
swift build
swift test
python3 -m py_compile Scripts/*.py
bash -n Scripts/*.sh
```

Local model-artifact validation:

```bash
Scripts/verify_local.sh
```

Optional broader local validation:

```bash
Scripts/verify_local.sh --all-inspects --demo-builds
```

`--all-inspects` loads all standard converted variants through the CLI. `--demo-builds` also builds the Catalyst and generic iOS Simulator demo targets. iOS Simulator can validate UI/build behavior, but model execution is rejected because MLX Swift does not currently support this model path in Simulator.

Physical-device staging helpers:

```bash
Scripts/stage_models_for_ios.sh --output /tmp/moebius-ipad-models ft_places2-q4
Scripts/run_device_autorun.sh --variant ft_places2-q4
Scripts/run_device_autorun.sh --model ft_places2 --compare
```

## Current Limitations

- MLX model execution requires supported Apple hardware. iOS Simulator execution is rejected before MLX initialization.
- Native model inference is fixed to `512x512` square inputs and outputs for the released checkpoints. Arbitrary photo sizes require crop, letterbox, or tile preprocessing outside the model.
- Model artifacts are distributed separately from this source repository.
- Lower-bit variants are measured tradeoffs, not automatic quality-equivalent replacements for f32.
- The 200 MB q4 candidate reaches the package-size target but has a larger measured fidelity drop.
- ONNX GPU acceleration depends on the ONNX Runtime providers available on the user’s machine. On macOS, CoreML provider coverage can be partial and may be slower than expected for this graph.
- The API should be treated as pre-1.0 until the first tagged release.
