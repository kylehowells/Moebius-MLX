# Moebius MLX ft_places2 onnx steps=20

This is a supplemental historical report for the demo-visible MLX variants against an ONNX-generated latent fixture. The current public cross-runtime benchmark is in [../ft_places2-cross-runtime-steps20](../ft_places2-cross-runtime-steps20/).

Source TSV: `benchmark.tsv`

| variant | model MB | max abs | mean abs | RMSE | p99 abs | PSNR dB | cosine | inference s | load s | total s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ft_places2-f32 | 1239.58 | 0.000272334 | 1.73542e-06 | 5.94996e-06 | 1.0103e-05 | 114.394 | 1.00000000 | 26.777 | 0.420 | 27.197 |
| ft_places2-f16 | 619.89 | 1.83539 | 0.0930781 | 0.165905 | 0.73437 | 25.487 | 0.98042941 | 20.783 | 0.285 | 21.069 |
| ft_places2-q8 | 600.23 | 1.83764 | 0.0928631 | 0.165784 | 0.738405 | 25.494 | 0.98044908 | 20.021 | 0.280 | 20.301 |
| ft_places2-q4 | 475.67 | 1.95124 | 0.0951408 | 0.176075 | 0.786058 | 24.971 | 0.97774184 | 18.039 | 0.329 | 18.368 |
| ft_places2-mxfp8 | 599.56 | 1.74931 | 0.0901569 | 0.164743 | 0.717825 | 25.548 | 0.98046005 | 21.273 | 0.277 | 21.550 |
| ft_places2-mxfp4 | 589.09 | 2.52835 | 0.0962882 | 0.184187 | 0.811111 | 24.579 | 0.97599477 | 20.942 | 0.273 | 21.216 |
| ft_places2-q4-ff-spatial-vae | 322.50 | 1.96471 | 0.0968293 | 0.17823 | 0.803101 | 24.865 | 0.97694558 | 17.184 | 0.345 | 17.529 |
| ft_places2-q4-all-spatial-vae | 200.50 | 2.14852 | 0.158999 | 0.254395 | 0.981538 | 21.774 | 0.95212305 | 17.438 | 0.255 | 17.693 |

- Lowest RMSE: `ft_places2-f32` at `5.94996e-06`.
- Fastest inference: `ft_places2-q4-ff-spatial-vae` at `17.184s`.
- Smallest package: `ft_places2-q4-all-spatial-vae` at `200.50 MB`.

Generated charts:

- `fidelity_errors.png`
- `performance_timings.png`
- `model_size.png`
- `speed_fidelity_tradeoff.png`
- `size_fidelity_tradeoff.png`
