# Benchmark Scripts

The main public benchmark entry point is:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements-benchmarks.txt

./.venv/bin/python Scripts/run_cross_runtime_benchmark.py \
  --output-dir Docs/BenchmarkReports/ft_places2-cross-runtime-steps20
```

The script compares:

- PyTorch reference implementation from the original Moebius code and checkpoint, on CPU and MPS where available.
- ONNX export from `simonw/Moebius-ONNX`, on CPU and CoreML where available.
- MLX variants from the `MoebiusMLX` Swift package.

It writes:

- `benchmark.csv`: raw benchmark rows.
- `summary.md`: results table and report.
- `fidelity_errors.png`, `performance_timings.png`, `model_size.png`, `speed_fidelity_tradeoff.png`, `size_fidelity_tradeoff.png`.
- `latent_contact_sheet.png`.
- `reference_fixture/` and `latents/` raw tensors used to calculate the metrics.

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
- `--mlx-variants ft_places2-f32,ft_places2-f16,...`: comma-separated MLX variants.
- `--no-download`: fail instead of downloading missing public artifacts.

The older focused helpers are still available:

- `benchmark_fidelity.sh`: MLX latent fidelity against an existing PyTorch/ONNX fixture.
- `run_benchmark_report.sh`: graph report for `benchmark_fidelity.sh`.
- `benchmark_image.sh`: MLX image-output benchmark against another MLX model.
- `run_image_benchmark_report.sh`: graph report for `benchmark_image.sh`.
