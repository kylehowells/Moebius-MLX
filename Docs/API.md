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

## Model Resolution And Downloads

Use `MoebiusModel.load(from:)` when the app already has a model directory, for example a bundled resource, an app-managed download, or a user-picked folder:

```swift
let localDirectory = URL(fileURLWithPath: "/path/to/ft_places2-q4", isDirectory: true)
let model = try MoebiusModel.load(from: localDirectory)
```

Use `MoebiusModel.fromPretrained(...)` when the model may need to be resolved or downloaded:

```swift
let model = try await MoebiusModel.fromPretrained(
	"ft_places2-q4",
	onDownloadProgress: { progress in
		Task { @MainActor in
			progressView.progress = Float(progress.fractionCompleted)
			statusLabel.text = "\(progress.completedUnitCount) / \(progress.totalUnitCount)"
		}
	}
)
```

`fromPretrained` accepts:

- a local model directory path
- a direct path to `manifest.json`
- a known variant name such as `ft_places2-q4`
- a Hugging Face repo id such as `iky1e/moebius-ft-places2-mlx-q4`
- a Hugging Face URL such as `https://huggingface.co/iky1e/moebius-ft-places2-mlx-q4`

Named variants map to one Hub repo per family and precision: `moebius-<family>-mlx-<precision>`. For example, `ft_places2-q4-all-spatial-vae` maps to `iky1e/moebius-ft-places2-mlx-q4-all-spatial-vae`.

Download progress is exposed through `MoebiusDownloadProgress`, which includes `repoID`, `revision`, `completedUnitCount`, `totalUnitCount`, `fractionCompleted`, and optional `bytesPerSecond`. The callback is passed through from the Hugging Face snapshot downloader so callers can show a determinate progress UI where the server reports total bytes.

Default materialized snapshot locations:

| Platform | Default |
|---|---|
| macOS with `HF_HUB_CACHE` | `$HF_HUB_CACHE/moebius-mlx` |
| macOS with `HF_HOME` | `$HF_HOME/hub/moebius-mlx` |
| macOS fallback | `~/.cache/huggingface/hub/moebius-mlx` |
| iOS / sandboxed app fallback | app `Library/Application Support/Moebius-MLX/HuggingFace` |

Pass `downloadBase:` to override where the materialized snapshot directory is written. The underlying Hub content cache is still created by Swift Hugging Face's default cache policy.

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

### Image Size Contract

`MoebiusImageInput` intentionally accepts only the released Moebius checkpoint's native working shape:

| Tensor | Shape |
|---|---|
| Source image | `[1, 512, 512, 3]` |
| Mask | `[1, 512, 512, 1]` |
| Latent | `[1, 64, 64, 4]` |
| UNet input | `[1 or 2, 64, 64, 9]` |

The fixed square shape comes from the original model configuration and trained lambda cross-attention relative-position tensors. The MLX package implements the released checkpoint behavior; it does not add arbitrary native image sizes or aspect ratios.

If your app accepts non-square images, convert them into the `512 x 512` model workspace first. Common choices are center crop, letterbox, or tiled inpainting. Keep the mask in the same transformed coordinate space, then map the generated square output back into your app's image coordinates.

## Core Types

`MoebiusModelResolver`

- `resolve(name:preferredDirectory:modelRoot:)`: finds a variant directory by explicit directory, model root, `MOEBIUS_MLX_MODEL_DIR`, `MOEBIUS_MODEL_ROOT`, app bundle resources, Application Support, or Documents.
- `resolveOrDownload(source:revision:preferredDirectory:modelRoot:hfToken:downloadBase:progress:onDownloadProgress:)`: resolves local paths and known model names first, then downloads from Hugging Face when needed.
- `downloadFromHub(repoID:revision:hfToken:downloadBase:onDownloadProgress:)`: downloads a specific Hub repo snapshot containing `README.md`, `manifest.json`, and `*.safetensors`.
- `defaultHubRepoID(for:)`: maps local variant names to the default `iky1e/moebius-...` Hub repo naming convention.

`MoebiusModel`

- `fromPretrained(_:revision:hfToken:downloadBase:progress:onDownloadProgress:)`: resolves or downloads a model, exposes download progress, then loads it.
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
| `inspect` | Load a converted model and print manifest, precision, active deployment files, size, and dtype byte-share details. |
| `download` | Resolve or download a model from Hugging Face without loading MLX weights. |
| `export-quantized` | Materialize quantized UNet and optional quantized VAE files inside a model directory. |
| `vae-smoke` | Run VAE encode/decode on a zero image tensor and print shapes. |
| `u-net-smoke` | Run one CFG UNet prediction on zero latent inputs and print the output shape. |
| `image-smoke` | Run the full image pipeline on a synthetic sample and print output stats. |
| `generate-image` | Run one file-backed image/mask pair through a converted model and write a PNG. |
| `memory-profile` | Run one image pipeline pass while logging RSS and MLX active/cache/peak memory at each stage. |
| `quality-check` | Compare normal and low-memory image-pipeline outputs for the same fixed input and tolerances. |
| `compare-image` | Run a deterministic sample across precision variants and compare image outputs. |
| `benchmark-image` | Compare synthetic or file-backed image output against a reference model. |
| `benchmark-fixture` | Compare MLX final latents against a deterministic PyTorch/ONNX fixture. |
| `u-net-debug` | Print early UNet debug stats for the zero-input parity case. |
| `verify-fixture` | Run the zero-latent UNet fixture and optionally diff against a raw f32 reference tensor. |

Examples:

```bash
swift run moebius-mlx inspect ft_places2-q4

swift run moebius-mlx download ft_places2-q4

swift run moebius-mlx download https://huggingface.co/iky1e/moebius-ft-places2-mlx-q4 \
  --download-base /tmp/moebius-hf-snapshots

swift run moebius-mlx image-smoke ft_places2-f16 --steps 2

swift run moebius-mlx quality-check ft_places2-q4-all-spatial-vae \
  --steps 1 \
  --seed 7 \
  --sample gradient-circle \
  --max-abs-tolerance 0 \
  --mean-abs-tolerance 0

swift run moebius-mlx memory-profile ft_places2-q4-all-spatial-vae \
  --steps 1 \
  --low-memory \
  --cache-limit-mb 20 \
  --reset-peak-at-stages

MOEBIUS_MLX_SERIAL_CFG=1 swift run moebius-mlx memory-profile ft_places2-q4-all-spatial-vae \
  --steps 8 \
  --low-memory \
  --cache-limit-mb 20 \
  --reset-peak-at-stages

When `--spatial-conv-chunk-rows` is omitted, the runtime uses the mobile default: `128` rows for batched classifier-free guidance and `64` rows for serial CFG. Use the option or `MOEBIUS_MLX_SPATIAL_CONV_CHUNK_ROWS` to sweep a specific device.

swift run moebius-mlx compare-image ft_places2 --steps 2

swift run moebius-mlx benchmark-image ft_places2-q4 \
  --reference-model ft_places2-f16 \
  --reference-name f16-image \
  --steps 2 \
  --samples gradient-circle

swift run moebius-mlx benchmark-fixture ft_places2-f32 \
  --fixture Fixtures/ft_places2-onnx-steps20 \
  --reference-name onnx
```

Generate the checked-in graph report:

```bash
Scripts/run_benchmark_report.sh \
  --fixture Fixtures/ft_places2-onnx-steps20 \
  --reference onnx \
  --steps 20 \
  --variants 'ft_places2-f32=ft_places2-f32,ft_places2-f16=ft_places2-f16,ft_places2-q8=ft_places2-q8,ft_places2-q4=ft_places2-q4,ft_places2-mxfp8=ft_places2-mxfp8,ft_places2-mxfp4=ft_places2-mxfp4,ft_places2-q4-ff-spatial-vae=ft_places2-q4-ff-spatial-vae,ft_places2-q4-all-spatial-vae=ft_places2-q4-all-spatial-vae' \
  --output-dir Docs/BenchmarkReports/ft_places2-onnx-steps20-visible-variants
```

## UIKit Demo Integration

The demo app uses the same `MoebiusPipeline.runImageRGBFloats` API as the CLI. It resolves model variants from bundle resources, Application Support, Documents, `MOEBIUS_MLX_MODEL_DIR`, `MOEBIUS_MODEL_ROOT`, and standard Hugging Face materialized snapshot locations. `MOEBIUS_MODEL_ROOT` is kept as a demo/autorun compatibility alias for the shared resolver's `MOEBIUS_MLX_MODEL_DIR`. The Mac Catalyst autorun path is controlled by:

```text
MOEBIUS_DEMO_AUTORUN=1
MOEBIUS_DEMO_COMPARE=1
MOEBIUS_DEMO_MODEL=ft_places2
MOEBIUS_DEMO_PRECISION=f16
MOEBIUS_DEMO_STEPS=2
MOEBIUS_DEMO_AUTORUN_STATUS=/tmp/moebius-demo-status.txt
```

Single-variant autorun status lines include `seconds=`, `model_mb=`, `memory_mb=`, and image min/max/mean. Comparison status lines include per-variant `model_mbs=`, `memory_mbs=`, and `seconds=` maps. The physical-device helper copies the same status file back from the app container.

The interactive UIKit surface follows the web-demo workflow: load a source image, paint the inpaint mask directly over it, pick a model from a menu, and generate into the result view. Advanced controls sit in the right inspector: model/precision selection, steps, guidance, seed, paste-back, brush size, paint/erase, sample reset, clear mask, and quant comparison. On narrow devices the inspector opens as a slide-over from the Options button. Imported images start with an empty editable mask; the synthetic sample starts with a circular mask for immediate one-click runs.

Physical-device staging uses:

```bash
Scripts/stage_models_for_ios.sh --output /tmp/moebius-ipad-models ft_places2-q4
Scripts/run_device_autorun.sh --variant ft_places2-q4
Scripts/run_device_autorun.sh --model ft_places2 --compare
Scripts/run_device_autorun.sh --model ft_places2 --compare --dry-run
Scripts/run_device_autorun.sh --model ft_places2 --compare --wait-seconds 300 --launch-wait-seconds 300 --autorun-timeout-seconds 900
```

The device autorun helper exits before build/install if CoreDevice reports the configured iPad as unavailable. `--wait-seconds` turns that into a bounded poll before build/install. `--launch-wait-seconds` retries the app launch if SpringBoard rejects it because the iPad is locked. It builds with `generic/platform=iOS` by default because CoreDevice ids are not the same ids shown by `xcodebuild -showdestinations`; the CoreDevice id is still used for install, model copy, launch, and status readback. After launch, the helper tags the run with `MOEBIUS_DEMO_AUTORUN_RUN_ID` and polls the app-container status file until that same run writes `done` or `error`, bounded by `--autorun-timeout-seconds`. `--compare` stages every visible precision for the selected model and launches the same comparison path as the Catalyst verifier. `--dry-run` prints the resolved staging list and launch environment before device lookup, so the compare plan can be validated while the physical iPad is offline.
