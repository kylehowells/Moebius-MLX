# Benchmark Reports

Checked-in reports live here so GitHub renders each benchmark folder with its table, charts, reproduction notes, and raw data pointers.

## Reports

- [ft_places2-cross-runtime-steps20](ft_places2-cross-runtime-steps20/): current portable report comparing PyTorch, ONNX, and MLX variants. This is the report to cite for public runtime, size, and fidelity comparisons.
- [ft_places2-onnx-steps20-visible-variants](ft_places2-onnx-steps20-visible-variants/): older focused report comparing demo-visible MLX variants against an ONNX-generated latent fixture. Kept as supplemental quantization history, not as the main public benchmark.
- [f32-mlx-speed-iteration](f32-mlx-speed-iteration.md): notes from the MLX f32 optimization pass that reduced the f32 fixture from roughly 19s to roughly 11s while preserving PyTorch parity.

## Regeneration

Use [Scripts/run_cross_runtime_benchmark.py](../../Scripts/run_cross_runtime_benchmark.py) for the current report:

```bash
./.venv/bin/python Scripts/run_cross_runtime_benchmark.py \
  --output-dir Docs/BenchmarkReports/ft_places2-cross-runtime-steps20
```

Focused MLX-only reports can still be regenerated with [Scripts/run_benchmark_report.sh](../../Scripts/run_benchmark_report.sh), which writes a `README.md` plus chart images in the selected output directory.
