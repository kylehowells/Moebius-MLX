# Moebius-MLX Docs

This directory is the public documentation index for the Swift MLX library, demo app, conversion tooling, and benchmark reports.

## Files

- [API.md](API.md): Swift API, CLI usage, model loading, download behavior, and device/demo notes.
- [PROJECT.md](PROJECT.md): architecture, conversion decisions, quantization notes, benchmark interpretation, demo behavior, and current limitations.
- [BenchmarkReports](BenchmarkReports/): checked-in benchmark outputs with CSV/TSV data, charts, and report pages.

## Current Benchmark Report

The canonical public benchmark report is [ft_places2-cross-runtime-steps20](BenchmarkReports/ft_places2-cross-runtime-steps20/). It compares PyTorch, ONNX, and MLX rows with the portable benchmark script in [Scripts/run_cross_runtime_benchmark.py](../Scripts/run_cross_runtime_benchmark.py).

The older [ft_places2-onnx-steps20-visible-variants](BenchmarkReports/ft_places2-onnx-steps20-visible-variants/) report is retained as supplemental history for the visible MLX quantization variants against an ONNX latent fixture.
