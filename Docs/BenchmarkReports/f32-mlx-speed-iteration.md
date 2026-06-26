# MLX f32 Speed Iteration

This note records the June 2026 f32-only optimization pass for the MLX runtime. The goal was to improve MLX f32 latency while preserving parity with the PyTorch MPS reference fixture.

## Fixture

- Model: `ft_places2-f32`
- Reference: PyTorch MPS
- Steps: 20 requested, 19 denoising steps after `strength=0.99`
- Guidance: `2.0`
- Benchmark path: `benchmark-fixture` latent fixture, not full image encode/decode
- Image/mask: repo-local Places fixture under `TestFixtures/places/`

## Result

| Runtime | Inference seconds | Load seconds | Total seconds | RMSE vs PyTorch MPS | Max abs | P99 abs | PSNR | Cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch MPS reference | 14.119 | 2.462 | 16.581 | 0.0 | 0.0 | 0.0 | inf | 1.00000007 |
| MLX f32 before optimization | 19.353 | 0.271 | 19.624 | 0.000005856 | 0.000354469 | 0.000009120 | 114.532 | 1.00000000 |
| MLX f32 after optimization | 9.215 | 0.490 | 9.705 | 0.000000729 | 0.000031292 | 0.000002146 | 131.622 | 1.00000000 |

The optimized MLX path is about 1.53x faster than PyTorch MPS on this fixture and about 2.10x faster than the original MLX path, while staying within floating-point noise against the same PyTorch reference.

## Changes

- Replaced the self-lambda positional convolution layout with an equivalent grouped convolution. The original implementation reshaped value channels into the batch dimension and ran many 1-channel convolutions. The optimized path keeps value channels as grouped input channels, tiles the shared positional weights and bias once, then runs one grouped convolution.
- Added per-generation cross-attention conditioning caches. The key/value-derived cross-attention tensors depend on the fixed Moebius conditioning IDs, not on the denoising latent, so they are prepared once per generation and reused across timesteps.

## Verification

Commands run from the repo root:

```bash
swift test
./.venv/bin/python Scripts/run_cross_runtime_benchmark.py \
  --output-dir BenchmarkArtifacts/f32-speed-iteration-optimized \
  --torch-devices mps \
  --onnx-providers '' \
  --mlx-variants ft_places2-f32
```

The checked benchmark artifacts for this local run are under `BenchmarkArtifacts/f32-speed-iteration-optimized/`.
