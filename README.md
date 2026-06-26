# Moebius-MLX

Swift and MLX port of Moebius image inpainting for Apple Silicon macOS, Mac Catalyst, and iPadOS.

`MoebiusMLX` loads converted Moebius model artifacts, downloads published variants from Hugging Face when needed, and exposes a small Swift API for image inpainting. The repository also includes a command-line tool, conversion and benchmark scripts, and a UIKit demo app.

## Contents

- `Sources/MoebiusMLX`: SwiftPM library product.
- `Sources/MoebiusMLXCLI`: `moebius-mlx` command-line tool.
- `Demo/MoebiusDemo`: UIKit demo app for Mac Catalyst and iPadOS.
- `Scripts`: conversion, validation, staging, and benchmark helpers.
- `Docs/README.md`: documentation index.
- `Docs/API.md`: detailed API and CLI notes.
- `Docs/PROJECT.md`: implementation notes, benchmark results, and current limitations.
- `Docs/BenchmarkReports`: checked-in benchmark CSV/TSV data, reports, and charts.

## Requirements

- Swift 6.1 or newer.
- Xcode with MLX-compatible Apple platform SDKs.
- Apple Silicon Mac for local MLX execution.
- macOS 14+ or iOS 17+.

MLX model execution is not available in iOS Simulator. Build the app in Simulator to check UI, then run inference on macOS, Mac Catalyst, or a real iPad.

## Install With SwiftPM

In Xcode, use **File > Add Package Dependencies** and add:

```text
https://github.com/kylehowells/Moebius-MLX.git
```

Then link the `MoebiusMLX` product to your app target.

For a package manifest:

```swift
// swift-tools-version: 6.1
import PackageDescription

let package = Package(
	name: "MyApp",
	platforms: [
		.macOS(.v14),
		.iOS(.v17),
	],
	dependencies: [
		.package(url: "https://github.com/kylehowells/Moebius-MLX.git", branch: "main"),
	],
	targets: [
		.target(
			name: "MyApp",
			dependencies: [
				.product(name: "MoebiusMLX", package: "Moebius-MLX"),
			]
		),
	]
)
```

After the first tagged release, prefer a version requirement such as `.upToNextMinor(from: "0.1.0")`.

## Quick Start

```swift
import MoebiusMLX

let model = try await MoebiusModel.fromPretrained(
	"ft_places2-q4",
	onDownloadProgress: { progress in
		print("download", progress.fractionCompleted)
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
	options: options,
	progress: { progress in
		if let step = progress.step, let total = progress.total {
			print("\(progress.stage) \(step)/\(total)")
		}
	}
)
```

`imageFloats` must be `[1, 512, 512, 3]` RGB in NHWC order normalized to `[-1, 1]`. `maskFloats` must be `[1, 512, 512, 1]`, where `1` marks the region to inpaint. `runImageRGBFloats` returns `[1, 512, 512, 3]` RGB values clamped to `[0, 1]`.

## Model Loading

`MoebiusModel.fromPretrained(_:)` accepts:

- a short variant name such as `ft_places2-q4`
- a Hugging Face repo id such as `iky1e/moebius-ft-places2-mlx-q4`
- a Hugging Face model URL
- a local model directory
- a direct path to `manifest.json`

Named variants map to one Hugging Face repo per model family and precision. For example:

```swift
let model = try await MoebiusModel.fromPretrained("ft_places2-q4")
let sameModel = try await MoebiusModel.fromPretrained("iky1e/moebius-ft-places2-mlx-q4")
```

Use `downloadBase:` when an app wants to control where materialized model snapshots are stored. On macOS, the default follows Hugging Face cache conventions under `HF_HUB_CACHE`, `HF_HOME`, or `~/.cache/huggingface/hub/moebius-mlx`. On iOS and Mac Catalyst sandboxed builds, the default materialized snapshot location is app-specific Application Support.

For bundled or self-hosted models:

```swift
let bundledURL = Bundle.main.url(forResource: "ft_places2-q4", withExtension: nil)!
let model = try MoebiusModel.load(from: bundledURL)
```

## Published Variants

The converted model set has four model families:

- `pretrained`: base Moebius checkpoint.
- `ft_places2`: inpainting model fine-tuned on Places2.
- `ft_celebahq`: inpainting model fine-tuned on CelebA-HQ faces.
- `ft_ffhq`: inpainting model fine-tuned on FFHQ faces.

Each family has these standard precision variants:

- `f32`
- `f16`
- `q8`
- `q4`
- `mxfp8`
- `mxfp4`

`ft_places2` also has two experimental q4 deployment-size candidates:

- `ft_places2-q4-ff-spatial-vae`: about 322 MB.
- `ft_places2-q4-all-spatial-vae`: about 200 MB.

The smaller candidates are useful for size experiments, but they trade away more fidelity than the default q4 package. See `Docs/PROJECT.md` and the supplemental [ft_places2 ONNX-visible variant report](Docs/BenchmarkReports/ft_places2-onnx-steps20-visible-variants/) for measured tradeoffs.

## Command Line

Build and test the package:

```bash
swift build
swift test
```

Inspect a model:

```bash
swift run moebius-mlx inspect ft_places2-q4
```

Run a small image smoke test:

```bash
swift run moebius-mlx image-smoke ft_places2-f16 --steps 2
```

Compare visible precision variants for one family:

```bash
swift run moebius-mlx compare-image ft_places2 --steps 2
```

Run local verification checks:

```bash
Scripts/verify_local.sh
```

Most scripts use `Models/Moebius-MLX` by default for local converted artifacts. Set `MOEBIUS_MLX_MODEL_DIR` to point at a different model root.

## Benchmarks

The portable cross-runtime benchmark compares PyTorch, ONNX, and MLX variants and can download missing public artifacts:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements-benchmarks.txt
./.venv/bin/python Scripts/run_cross_runtime_benchmark.py
```

See [Scripts/README.md](Scripts/README.md) for benchmark options. The checked-in reference report is [Docs/BenchmarkReports/ft_places2-cross-runtime-steps20](Docs/BenchmarkReports/ft_places2-cross-runtime-steps20/).

## Demo App

The demo app is in `Demo/MoebiusDemo`.

It provides a WebGL-demo-style workflow: pick a model, load or use a sample image, paint the inpaint mask directly on the source image, generate, and compare precision variants. See [Demo/MoebiusDemo/README.md](Demo/MoebiusDemo/README.md) for screenshots, build instructions, and model provisioning notes.

## Project Status

The library and demo are usable for local experimentation and public review. Current caveats:

- MLX execution requires supported Apple hardware and does not run in iOS Simulator.
- Model artifacts are not committed to this repository.
- Lower-bit variants are still quality/performance tradeoff candidates, not drop-in f32 replacements.
- The API is small but should still be treated as pre-1.0 until a tagged release is cut.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.
