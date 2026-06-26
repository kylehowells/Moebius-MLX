#!/usr/bin/env python3
"""Generate deterministic Moebius PyTorch/ONNX latent fixtures.

The Swift CLI benchmark consumes this output with:

  swift run moebius-mlx benchmark-fixture ft_places2-f16 --fixture <out>

The fixture is intentionally latent-level. It fixes the image, mask, initial noise,
scheduler, guidance scale, and denoising steps, then writes:

  masked_lat.bin
  mask64.bin
  init_latents.bin
  final_latents.bin
  meta.json

All raw tensors are little-endian float32 in NCHW layout by default.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("MOEBIUS_SOURCE_ROOT", Path.cwd()))
CODE_ROOT = ROOT / "Moebius-Code"
ONNX_ROOT = ROOT / "Moebius-ONNX"
WEIGHTS_ROOT = ROOT / "Moebius-Models"
VAE_DIR = ROOT / "PixelHacker" / "vae"
DEFAULT_IMAGE = REPO_ROOT / "TestFixtures" / "places" / "places_desert_canyon.png"
DEFAULT_MASK = REPO_ROOT / "TestFixtures" / "places" / "places_desert_canyon_mask.png"
SCALING_FACTOR = 0.13025
NOISE_OFFSET = 0.0357
BETA_START = 0.00085
BETA_END = 0.012
TRAIN_TIMESTEPS = 1000


def require_module(name: str):
    try:
        return __import__(name)
    except ImportError as exc:
        raise SystemExit(
            f"Missing Python dependency: {name}. Install it before generating this fixture."
        ) from exc


def make_ddim(num_steps: int) -> tuple[np.ndarray, np.ndarray]:
    betas = np.linspace(BETA_START**0.5, BETA_END**0.5, TRAIN_TIMESTEPS, dtype=np.float64) ** 2
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    step_ratio = TRAIN_TIMESTEPS // num_steps
    timesteps = (np.arange(0, num_steps) * step_ratio).round()[::-1].astype(np.int64)
    return alphas_cumprod, timesteps


def ddim_step(
    model_output: np.ndarray,
    timestep: int,
    sample: np.ndarray,
    alphas_cumprod: np.ndarray,
    previous_timestep: int,
) -> np.ndarray:
    ac_t = alphas_cumprod[timestep]
    ac_prev = alphas_cumprod[previous_timestep] if previous_timestep >= 0 else 1.0
    beta_t = 1 - ac_t
    pred_x0 = (sample - np.sqrt(beta_t) * model_output) / np.sqrt(ac_t)
    pred_dir = np.sqrt(1 - ac_prev) * model_output
    return np.sqrt(ac_prev) * pred_x0 + pred_dir


def prepare_inputs(image_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = Image.open(image_path).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
    mask = Image.open(mask_path).convert("L").resize((512, 512), Image.Resampling.NEAREST)
    mask = mask.point(lambda value: 0 if value < 128 else 255, "L")

    image_hwc = np.asarray(image).astype(np.float32) / 255.0 * 2 - 1
    image_nchw = image_hwc.transpose(2, 0, 1)[None]
    mask_bin = (np.asarray(mask).astype(np.float32) / 255.0 >= 0.5).astype(np.float32)
    masked_image = image_nchw * (1 - mask_bin[None, None])
    return image_nchw, masked_image, mask_bin


def make_mask64(mask_bin: np.ndarray, latent_size: int = 64) -> np.ndarray:
    mask_img = Image.fromarray((mask_bin * 255).astype(np.uint8))
    mask_small = mask_img.resize((latent_size, latent_size), Image.Resampling.NEAREST)
    return (np.asarray(mask_small).astype(np.float32) / 255.0)[None, None]


def make_noise(seed: int, shape: tuple[int, int, int, int]) -> np.ndarray:
    torch = require_module("torch")
    torch.manual_seed(seed)
    noise = torch.randn(*shape)
    noise = noise + NOISE_OFFSET * torch.randn(shape[0], shape[1], 1, 1)
    return noise.numpy().astype(np.float32)


def run_onnx(args, image_nchw: np.ndarray, masked_image: np.ndarray, mask64: np.ndarray, init_latents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ort = require_module("onnxruntime")
    enc = ort.InferenceSession(str(args.onnx_root / "vae_encoder.onnx"), providers=["CPUExecutionProvider"])
    unet = ort.InferenceSession(str(args.onnx_root / "unet.onnx"), providers=["CPUExecutionProvider"])

    masked_lat = enc.run(None, {"image": masked_image})[0][:, :4] * SCALING_FACTOR
    alphas_cumprod, timesteps = make_ddim(args.steps)
    timesteps = timesteps[max(args.steps - int(args.steps * args.strength), 0) :]
    latents = init_latents.copy()
    input_ids = np.stack([np.arange(10, 20), np.arange(0, 10)]).astype(np.int64)

    for index, timestep in enumerate(timesteps):
        previous = timesteps[index + 1] if index + 1 < len(timesteps) else -1
        nine = np.concatenate([latents, mask64, masked_lat], axis=1)
        nine2 = np.concatenate([nine, nine], axis=0).astype(np.float32)
        pred = unet.run(
            None,
            {
                "latent": nine2,
                "timesteps": np.array([timestep, timestep], dtype=np.int64),
                "input_ids": input_ids,
            },
        )[0]
        uncond, cond = pred[0:1], pred[1:2]
        cfg = uncond + args.guidance * (cond - uncond)
        latents = ddim_step(cfg, int(timestep), latents, alphas_cumprod, int(previous)).astype(np.float32)
    return masked_lat.astype(np.float32), latents.astype(np.float32)


def run_torch(args, image_nchw: np.ndarray, masked_image: np.ndarray, mask64: np.ndarray, init_latents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    torch = require_module("torch")
    require_module("diffusers")

    sys.path.insert(0, str(args.code_root))
    cwd = os.getcwd()
    os.chdir(args.code_root)
    try:
        # model_lib.__init__ imports the PixelHacker teacher GLA module, which depends on
        # CUDA-only flash-linear-attention. Moebius uses the student Lambda-DWConv model,
        # so provide a dummy teacher module and keep the actual student import path intact.
        if "model_lib.nets.unet_gla" not in sys.modules:
            dummy_unet_gla = types.ModuleType("model_lib.nets.unet_gla")
            dummy_unet_gla.UNet2DGLAConditionModel = None
            sys.modules["model_lib.nets.unet_gla"] = dummy_unet_gla

        from diffusers import AutoencoderKL
        from removal.v1_2.removal_model import build_removal_model, load_cfg, load_removal_model

        cfg = load_cfg(str(args.code_root / "config" / "model_cfg" / "moebius.yaml"))
        cfg["vae"]["model_dir"] = str(args.vae_dir)
        removal_model = build_removal_model(cfg, 20).eval()
        load_removal_model(removal_model, str(args.weights_root / args.model / "diffusion_pytorch_model.bin"), "cpu")
        vae = AutoencoderKL.from_pretrained(str(args.vae_dir)).eval()

        alphas_cumprod, timesteps = make_ddim(args.steps)
        timesteps = timesteps[max(args.steps - int(args.steps * args.strength), 0) :]
        latents = torch.tensor(init_latents)
        input_ids = torch.tensor(np.stack([np.arange(10, 20), np.arange(0, 10)]), dtype=torch.int64)
        mask_tensor = torch.tensor(mask64)

        with torch.no_grad():
            masked_lat = vae.encode(torch.tensor(masked_image)).latent_dist.mean * SCALING_FACTOR
            for index, timestep in enumerate(timesteps):
                previous = timesteps[index + 1] if index + 1 < len(timesteps) else -1
                nine = torch.cat([latents, mask_tensor, masked_lat], dim=1)
                nine2 = torch.cat([nine, nine], dim=0)
                ts = torch.tensor([int(timestep), int(timestep)], dtype=torch.int64)
                pred = removal_model(nine2, ts, input_ids).sample
                uncond, cond = pred[0:1], pred[1:2]
                cfg_pred = uncond + args.guidance * (cond - uncond)
                latents = torch.tensor(
                    ddim_step(
                        cfg_pred.numpy(),
                        int(timestep),
                        latents.numpy(),
                        alphas_cumprod,
                        int(previous),
                    ).astype(np.float32)
                )
            return masked_lat.numpy().astype(np.float32), latents.numpy().astype(np.float32)
    finally:
        os.chdir(cwd)


def metrics(values: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    error = np.abs(values - reference)
    rmse = float(np.sqrt(np.mean(error * error)))
    peak = max(float(np.abs(reference).max()), 1.0)
    return {
        "max_abs": float(error.max()),
        "mean_abs": float(error.mean()),
        "rmse": rmse,
        "p99_abs": float(np.quantile(error, 0.99)),
        "psnr_db": float("inf") if rmse == 0 else float(20 * np.log10(peak / rmse)),
        "cosine_similarity": float(np.vdot(values.ravel(), reference.ravel()) / (np.linalg.norm(values.ravel()) * np.linalg.norm(reference.ravel()))),
    }


def write_fixture(args, masked_lat: np.ndarray, mask64: np.ndarray, init_latents: np.ndarray, final_latents: np.ndarray, reference: str) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    masked_lat.astype("<f4").tofile(args.output / "masked_lat.bin")
    mask64.astype("<f4").tofile(args.output / "mask64.bin")
    init_latents.astype("<f4").tofile(args.output / "init_latents.bin")
    final_latents.astype("<f4").tofile(args.output / "final_latents.bin")
    meta = {
        "reference": reference,
        "model": args.model,
        "steps": args.steps,
        "guidance": args.guidance,
        "strength": args.strength,
        "seed": args.seed,
        "noise_offset": NOISE_OFFSET,
        "scaling_factor": SCALING_FACTOR,
        "layout": "nchw",
    }
    (args.output / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ft_places2", choices=["pretrained", "ft_places2", "ft_celebahq", "ft_ffhq"])
    parser.add_argument("--reference", default="onnx", choices=["onnx", "pytorch"])
    parser.add_argument("--compare", choices=["onnx", "pytorch"])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--strength", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--output", type=Path, default=ROOT / "Moebius-MLX-fixtures" / "ft_places2-onnx")
    parser.add_argument("--code-root", type=Path, default=CODE_ROOT)
    parser.add_argument("--onnx-root", type=Path, default=ONNX_ROOT)
    parser.add_argument("--weights-root", type=Path, default=WEIGHTS_ROOT)
    parser.add_argument("--vae-dir", type=Path, default=VAE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_nchw, masked_image, mask_bin = prepare_inputs(args.image, args.mask)
    mask64 = make_mask64(mask_bin)
    init_latents = make_noise(args.seed, (1, 4, 64, 64))

    runners = {
        "onnx": run_onnx,
        "pytorch": run_torch,
    }
    masked_lat, final_latents = runners[args.reference](args, image_nchw, masked_image, mask64, init_latents)
    write_fixture(args, masked_lat, mask64, init_latents, final_latents, args.reference)
    print(f"wrote fixture: {args.output}")
    print(f"reference: {args.reference}")
    print("final_latents mean=%.6f std=%.6f" % (final_latents.mean(), final_latents.std()))

    if args.compare:
        _, compare_latents = runners[args.compare](args, image_nchw, masked_image, mask64, init_latents)
        comparison = metrics(compare_latents, final_latents)
        comparison["reference"] = args.reference
        comparison["compare"] = args.compare
        comparison["steps"] = args.steps
        comparison["guidance"] = args.guidance
        comparison["strength"] = args.strength
        (args.output / f"{args.compare}_vs_{args.reference}_metrics.json").write_text(json.dumps(comparison, indent=2) + "\n")
        print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
