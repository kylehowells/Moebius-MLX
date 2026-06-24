# Moebius-MLX API And CLI Reference

This package exposes a Swift library product and a command-line tool:

```swift
.library(name: "MoebiusMLX", targets: ["MoebiusMLX"])
.executable(name: "moebius-mlx", targets: ["MoebiusMLXCLI"])
```

Supported platforms are macOS 14 and iOS 17. Model execution is supported on Apple Silicon macOS, Mac Catalyst, and physical iOS devices. The library rejects iOS Simulator model execution before touching MLX because MLX Swift currently aborts while initializing its Metal stream/device there.

## Model Layout

Model variants are resolved by name or directory. A variant directory contains:

```text
manifest.json
unet.safetensors
vae_encoder.safetensors
vae_decoder.safetensors
```

Quantized runtime packages may also contain:

```text
unet_quantized.safetensors
unet_quantized_pointwise_feedforward.safetensors
unet_quantized_pointwise_feedforward_spatial.safetensors
unet_quantized_pointwise_all_spatial.safetensors
vae_encoder_quantized.safetensors
vae_decoder_quantized.safetensors
```

The manifest decides which files are active. `MoebiusModelDeploymentInfo` reports the manifest-selected deployment size, following symlinks when candidate views are used.

## Swift Usage

```swift
import MoebiusMLX

let directory = try MoebiusModelResolver.resolve(name: "ft_places2-q4")
let model = try MoebiusModel.load(from: directory) { stage in
	print(stage)
}
let pipeline = MoebiusPipeline(model: model)

let input = MoebiusImageInput(
	imageNHWC: imageFloats512,
	maskNHWC: maskFloats512
)
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

`imageNHWC` is `512 * 512 * 3` floats in `[-1, 1]`. `maskNHWC` is `512 * 512` floats where `1` means inpaint. The output is `512 * 512 * 3` floats in `[0, 1]`.

## Core Types

`MoebiusModelResolver`

- `resolve(name:preferredDirectory:modelRoot:)`: finds a variant directory by explicit directory, model root, `MOEBIUS_MLX_MODEL_DIR`, `MOEBIUS_MODEL_ROOT`, app bundle resources, Application Support, Documents, or the local development model path.

`MoebiusModel`

- `load(from:progress:)`: loads manifest-selected UNet and VAE weights.
- `loadSourceUNet(from:)`: loads the source f16/f32 UNet for conversion/debug paths.
- `loadSourceAutoencoder(from:)`: loads source VAE files for conversion/debug paths.
- `applyQuantizationIfNeeded(...)`: applies quantized module replacement according to manifest precision and policies.

`MoebiusPipeline`

- `runImage(input:options:progress:)`: runs VAE encode, denoising, VAE decode, and optional paste-back.
- `runImageRGBFloats(input:options:progress:)`: convenience wrapper returning a Swift `[Float]`.
- `runLatentFixture(maskedLatents:mask:initLatents:options:)`: deterministic latent-level benchmark path.
- `predictNoiseCFG(latent:timestep:guidanceScale:)`: lower-level UNet CFG prediction.

`MoebiusModelDeploymentInfo`

- `load(from:)`: reads a manifest and reports active deployment bytes.
- `totalMB`, `totalMiB`, `unetMB`, `vaeMB`: user-facing package-size metrics.
- `files`: active UNet/VAE file entries with roles and byte counts.

`MoebiusRuntime`

- `supportsModelExecution`: false on iOS Simulator, true elsewhere.
- `validateModelExecutionSupport()`: throws a user-facing unsupported-runtime error when execution is unsafe.

## CLI

Run:

```bash
swift run moebius-mlx <subcommand>
```

Subcommands:

| Command | Purpose |
|---|---|
| `inspect` | Load a converted model and print manifest, precision, file, and deployment-size details. |
| `export-quantized` | Materialize quantized UNet and optional quantized VAE files inside a model directory. |
| `vae-smoke` | Run VAE encode/decode on a zero image tensor and print shapes. |
| `u-net-smoke` | Run one CFG UNet prediction on zero latent inputs and print the output shape. |
| `image-smoke` | Run the full image pipeline on a synthetic sample and print output stats. |
| `compare-image` | Run a deterministic sample across precision variants and compare image outputs. |
| `benchmark-image` | Compare synthetic or file-backed image output against a reference model. |
| `benchmark-fixture` | Compare MLX final latents against a deterministic PyTorch/ONNX fixture. |
| `u-net-debug` | Print early UNet debug stats for the zero-input parity case. |
| `verify-fixture` | Run the zero-latent UNet fixture and optionally diff against a raw f32 reference tensor. |

Examples:

```bash
swift run moebius-mlx inspect ft_places2-q4

swift run moebius-mlx image-smoke ft_places2-f16 --steps 2

swift run moebius-mlx compare-image ft_places2 --steps 2

swift run moebius-mlx benchmark-image ft_places2-q4 \
  --reference-model ft_places2-f16 \
  --reference-name f16-image \
  --steps 2 \
  --samples gradient-circle

swift run moebius-mlx benchmark-fixture ft_places2-f32 \
  --fixture /Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX-benchmarks/fixtures/ft_places2-onnx-steps20 \
  --reference-name onnx
```

Generate the checked-in graph report:

```bash
Scripts/run_benchmark_report.sh \
  --fixture /Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX-benchmarks/fixtures/ft_places2-onnx-steps20 \
  --reference onnx \
  --steps 20 \
  --variants 'ft_places2-f32=ft_places2-f32,ft_places2-f16=ft_places2-f16,ft_places2-q8=ft_places2-q8,ft_places2-q4=ft_places2-q4,ft_places2-mxfp8=ft_places2-mxfp8,ft_places2-mxfp4=ft_places2-mxfp4,ft_places2-q4-ff-spatial-vae=ft_places2-q4-ff-spatial-vae,ft_places2-q4-all-spatial-vae=ft_places2-q4-all-spatial-vae' \
  --output-dir Docs/BenchmarkReports/ft_places2-onnx-steps20-visible-variants
```

## UIKit Demo Integration

The demo app uses the same `MoebiusPipeline.runImageRGBFloats` API as the CLI. It resolves model variants from bundle resources, Application Support, Documents, `MOEBIUS_MLX_MODEL_DIR`, `MOEBIUS_MODEL_ROOT`, and the development model directory. `MOEBIUS_MODEL_ROOT` is kept as a demo/autorun compatibility alias for the shared resolver's `MOEBIUS_MLX_MODEL_DIR`. The Mac Catalyst autorun path is controlled by:

```text
MOEBIUS_DEMO_AUTORUN=1
MOEBIUS_DEMO_COMPARE=1
MOEBIUS_DEMO_MODEL=ft_places2
MOEBIUS_DEMO_PRECISION=f16
MOEBIUS_DEMO_STEPS=2
MOEBIUS_DEMO_AUTORUN_STATUS=/tmp/moebius-demo-status.txt
```

Physical-device staging uses:

```bash
Scripts/stage_models_for_ios.sh --output /tmp/moebius-ipad-models ft_places2-q4
Scripts/run_device_autorun.sh --variant ft_places2-q4
Scripts/run_device_autorun.sh --model ft_places2 --compare
Scripts/run_device_autorun.sh --model ft_places2 --compare --dry-run
Scripts/run_device_autorun.sh --model ft_places2 --compare --wait-seconds 300
```

The device autorun helper exits before build/install if CoreDevice reports the configured iPad as unavailable. `--wait-seconds` turns that into a bounded poll before build/install. `--compare` stages every visible precision for the selected model and launches the same comparison path as the Catalyst verifier. `--dry-run` prints the resolved staging list and launch environment before device lookup, so the compare plan can be validated while the physical iPad is offline.
