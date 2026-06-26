# Scripts

This directory contains the conversion, verification, model-staging, and benchmark helpers used by Moebius-MLX. The main public benchmark path is `run_cross_runtime_benchmark.py`; the older focused benchmark helpers are retained for MLX-only regression checks and historical reports.

## Public Benchmark

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements-benchmarks.txt

./.venv/bin/python Scripts/run_cross_runtime_benchmark.py \
  --output-dir Docs/BenchmarkReports/ft_places2-cross-runtime-steps20
```

`run_cross_runtime_benchmark.py` compares:

- PyTorch reference implementation from the original Moebius code and checkpoint, on CPU and MPS where available.
- ONNX export from `simonw/Moebius-ONNX`, on requested ONNX Runtime providers where available.
- MLX variants from the `MoebiusMLX` Swift package.

It writes:

- `benchmark.csv`: raw benchmark rows.
- `README.md`: report table, interpretation, and reproduction notes.
- `fidelity_errors.png`, `performance_timings.png`, `model_size.png`, `speed_fidelity_tradeoff.png`, `size_fidelity_tradeoff.png`.
- `latent_contact_sheet.png`.
- `reference_fixture/` and `latents/` raw tensors used to calculate the metrics.

The CSV and report table include every requested backend/device row. Chart images omit CPU baselines and rows above `--chart-max-total-seconds` by default, so slow diagnostic rows do not flatten the useful GPU/MLX comparisons.

By default, missing public artifacts are downloaded into `BenchmarkArtifacts/cross-runtime`, which is ignored by Git. You can use existing local artifacts instead:

```bash
./.venv/bin/python Scripts/run_cross_runtime_benchmark.py \
  --pytorch-code-root /path/to/Moebius-code \
  --pytorch-weights-root /path/to/hustvl-Moebius-weights \
  --vae-dir /path/to/PixelHacker/vae \
  --onnx-root /path/to/Moebius-ONNX \
  --mlx-model-root /path/to/Moebius-MLX-models
```

Useful options:

- `--torch-devices cpu,mps`: comma-separated PyTorch devices. Unavailable devices are skipped.
- `--onnx-providers CPUExecutionProvider,CoreMLExecutionProvider`: comma-separated ONNX Runtime providers. Unavailable providers are skipped.
- `--chart-max-total-seconds 60`: keep slower rows in the CSV/table but omit them from chart images.
- `--chart-include-devices MLX,mps`: optional comma-separated chart allow-list for row device/provider names.
- `--mlx-variants ft_places2-f32,ft_places2-f16,...`: comma-separated MLX variants.
- `--no-download`: fail instead of downloading missing public artifacts.

## Script Inventory

### Conversion And Model Packaging

- `convert_moebius.py`: Converts original Moebius PyTorch checkpoints into MLX `safetensors` artifacts and writes model manifests. This is the primary conversion script.
- `materialize_candidate_variants.py`: Builds candidate model variant folders from converted artifacts, including q4 deployment-size experiments.
- `audit_model_size.py`: Prints tensor-level and package-level size breakdowns for converted model directories.
- `stage_models_for_ios.sh`: Copies selected model variants into a smaller staging directory suitable for app bundle/device testing.
- `build_mlx_metallib.sh`: Builds or refreshes the MLX Metal library asset used by the demo/runtime packaging flow.

### Cross-Runtime Benchmarks

- `run_cross_runtime_benchmark.py`: Portable PyTorch/ONNX/MLX benchmark with Hugging Face/GitHub artifact fallback, CSV output, charts, contact sheet, and report generation. This is the benchmark to use for public comparisons.
- `generate_reference_fixture.py`: Creates deterministic image, mask, and latent fixture tensors used by benchmark scripts.

### Focused MLX Benchmark Helpers

- `benchmark_fidelity.sh`: Runs the Swift CLI latent benchmark for MLX variants against an existing reference fixture.
- `run_benchmark_report.sh`: Wraps `benchmark_fidelity.sh`, then calls `plot_benchmark_report.py` to produce a `README.md` report and charts.
- `plot_benchmark_report.py`: Plots a focused benchmark TSV into fidelity, runtime, size, and tradeoff charts.
- `benchmark_image.sh`: Runs the Swift CLI image-output benchmark for MLX variants against another MLX reference model.
- `run_image_benchmark_report.sh`: Wraps `benchmark_image.sh` and generates chart/report output for image-level MLX comparisons.
- `make_three_photo_contact_sheet.py`: Builds visual comparison sheets for selected output images and diff views.

### App And Device Validation

- `run_device_autorun.sh`: Builds/installs/runs the demo app with autorun arguments for iPadOS or Mac Catalyst device-style checks.
- `verify_local.sh`: Local verification bundle covering Swift package checks, CLI smoke tests, script syntax, checked-in report artifacts, optional model inspections, and optional demo builds.

## What Is Kept

No benchmark scripts are currently considered dead. The cross-runtime benchmark is the public path, while the focused MLX helpers are still useful for quick regression reports and for reproducing older checked-in benchmark history. Generated caches such as `.DS_Store`, `__pycache__`, `BenchmarkArtifacts/`, and top-level `BenchmarkReports/` are ignored and should not be committed.
