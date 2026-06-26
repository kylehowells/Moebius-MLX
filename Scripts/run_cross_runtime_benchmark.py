#!/usr/bin/env python3
"""Run portable PyTorch, ONNX, and MLX Moebius latent benchmarks.

This script is the public cross-runtime benchmark entry point. It can use
already-downloaded model directories, or bootstrap missing reference artifacts
from Hugging Face/GitHub into a local benchmark artifact folder.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = ROOT / "BenchmarkArtifacts" / "cross-runtime"
DEFAULT_OUTPUT_DIR = ROOT / "Docs" / "BenchmarkReports" / "ft_places2-cross-runtime-steps20"
MLX_VARIANTS = (
    "ft_places2-f32",
    "ft_places2-f16",
    "ft_places2-q8",
    "ft_places2-q4",
    "ft_places2-mxfp8",
    "ft_places2-mxfp4",
    "ft_places2-q4-ff-spatial-vae",
    "ft_places2-q4-all-spatial-vae",
)


def load_fixture_module():
    path = ROOT / "Scripts" / "generate_reference_fixture.py"
    spec = importlib.util.spec_from_file_location("generate_reference_fixture", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixture = load_fixture_module()


@dataclass
class RuntimeRow:
    name: str
    backend: str
    variant: str
    reference: str
    model_mb: float
    max_abs: float
    mean_abs: float
    rmse: float
    p99_abs: float
    psnr_db: float
    cosine: float
    inference_seconds: float
    load_seconds: float
    total_seconds: float
    device: str
    notes: str = ""


def require_module(name: str):
    try:
        return __import__(name)
    except ImportError as exc:
        raise SystemExit(
            f"Missing Python dependency: {name}. Install benchmark dependencies with:\n"
            f"  python3 -m pip install -r requirements-benchmarks.txt"
        ) from exc


def run_command(command: list[str], cwd: Path | None = None, capture: bool = False) -> str:
    if capture:
        return subprocess.check_output(command, cwd=cwd, text=True)
    subprocess.check_call(command, cwd=cwd)
    return ""


def git_clone_if_needed(repo: str, destination: Path, download_missing: bool) -> None:
    if destination.exists():
        return
    if not download_missing:
        raise SystemExit(f"Missing directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {repo} -> {destination}", file=sys.stderr)
    run_command(["git", "clone", "--depth", "1", repo, str(destination)])


def hf_snapshot_if_needed(repo_id: str, destination: Path, download_missing: bool, token: str | None = None, allow_patterns: list[str] | None = None) -> None:
    if destination.exists() and any(destination.iterdir()):
        return
    if not download_missing:
        raise SystemExit(f"Missing directory: {destination}")
    huggingface_hub = require_module("huggingface_hub")
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} -> {destination}", file=sys.stderr)
    huggingface_hub.snapshot_download(
        repo_id=repo_id,
        local_dir=destination,
        token=token,
        allow_patterns=allow_patterns,
    )


def ensure_reference_artifacts(args: argparse.Namespace) -> None:
    git_clone_if_needed(args.pytorch_code_repo, args.pytorch_code_root, args.download_missing)
    hf_snapshot_if_needed(args.pytorch_weights_repo, args.pytorch_weights_root, args.download_missing, token=args.hf_token)
    if not args.vae_dir.exists():
        hf_snapshot_if_needed(args.vae_repo, args.vae_dir.parent, args.download_missing, token=args.hf_token, allow_patterns=["vae/*"])
    hf_snapshot_if_needed(args.onnx_repo, args.onnx_root, args.download_missing, token=args.hf_token)

    required = [
        args.pytorch_code_root / "config" / "model_cfg" / "moebius.yaml",
        args.pytorch_weights_root / args.family / "diffusion_pytorch_model.bin",
        args.vae_dir / "config.json",
        args.vae_dir / "diffusion_pytorch_model.bin",
        args.onnx_root / "vae_encoder.onnx",
        args.onnx_root / "unet.onnx",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        joined = "\n".join(f"  {path}" for path in missing)
        raise SystemExit(f"Missing required benchmark artifacts:\n{joined}")


def directory_size_mb(path: Path, include: list[Path] | None = None) -> float:
    if include is not None:
        return sum(item.stat().st_size for item in include if item.exists()) / 1_000_000.0
    total = 0
    for item in path.rglob("*"):
        if item.is_file() and ".git" not in item.parts:
            total += item.stat().st_size
    return total / 1_000_000.0


def tensor_metrics(values: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    delta = (values.astype(np.float32) - reference.astype(np.float32)).reshape(-1)
    abs_delta = np.abs(delta)
    rmse = float(np.sqrt(np.mean(delta * delta)))
    peak = max(float(np.max(np.abs(reference))), 1.0)
    denom = float(np.linalg.norm(values.reshape(-1)) * np.linalg.norm(reference.reshape(-1)))
    return {
        "max_abs": float(np.max(abs_delta)),
        "mean_abs": float(np.mean(abs_delta)),
        "rmse": rmse,
        "p99_abs": float(np.quantile(abs_delta, 0.99)),
        "psnr_db": float("inf") if rmse == 0 else float(20.0 * math.log10(peak / rmse)),
        "cosine": 1.0 if denom == 0 else float(np.vdot(values.reshape(-1), reference.reshape(-1)) / denom),
    }


def write_raw_tensor(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values.astype("<f4").tofile(path)


def read_raw_tensor(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    return np.fromfile(path, dtype="<f4").reshape(shape)


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_"} else "-" for character in value)


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def available_torch_devices(requested: list[str]) -> list[str]:
    torch = require_module("torch")
    resolved: list[str] = []
    for device in requested:
        if device == "auto":
            resolved.append("mps" if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() else "cpu")
        elif device == "mps":
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                resolved.append(device)
            else:
                print("Skipping PyTorch MPS because it is not available on this machine.", file=sys.stderr)
        else:
            resolved.append(device)
    return list(dict.fromkeys(resolved))


def available_onnx_providers(requested: list[str]) -> list[str]:
    ort = require_module("onnxruntime")
    available = set(ort.get_available_providers())
    resolved: list[str] = []
    for provider in requested:
        if provider in available:
            resolved.append(provider)
        else:
            print(f"Skipping ONNX provider {provider}; available providers: {sorted(available)}", file=sys.stderr)
    return resolved


def make_reference_inputs(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image_nchw, masked_image, mask_bin = fixture.prepare_inputs(args.image, args.mask)
    mask64 = fixture.make_mask64(mask_bin)
    init_latents = fixture.make_noise(args.seed, (1, 4, 64, 64))
    return image_nchw, masked_image, mask64, init_latents


def run_onnx(
    args: argparse.Namespace,
    provider: str,
    masked_image: np.ndarray,
    mask64: np.ndarray,
    init_latents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    ort = require_module("onnxruntime")
    load_started = time.perf_counter()
    encoder = ort.InferenceSession(str(args.onnx_root / "vae_encoder.onnx"), providers=[provider])
    unet = ort.InferenceSession(str(args.onnx_root / "unet.onnx"), providers=[provider])
    load_seconds = time.perf_counter() - load_started

    run_started = time.perf_counter()
    masked_lat = encoder.run(None, {"image": masked_image})[0][:, :4] * fixture.SCALING_FACTOR
    alphas_cumprod, timesteps = fixture.make_ddim(args.steps)
    timesteps = timesteps[max(args.steps - int(args.steps * args.strength), 0) :]
    latents = init_latents.copy()
    input_ids = np.stack([np.arange(10, 20), np.arange(0, 10)]).astype(np.int64)
    for index, timestep in enumerate(timesteps):
        previous = timesteps[index + 1] if index + 1 < len(timesteps) else -1
        nine = np.concatenate([latents, mask64, masked_lat], axis=1)
        pred = unet.run(
            None,
            {
                "latent": np.concatenate([nine, nine], axis=0).astype(np.float32),
                "timesteps": np.array([timestep, timestep], dtype=np.int64),
                "input_ids": input_ids,
            },
        )[0]
        uncond, cond = pred[0:1], pred[1:2]
        cfg = uncond + args.guidance * (cond - uncond)
        latents = fixture.ddim_step(cfg, int(timestep), latents, alphas_cumprod, int(previous)).astype(np.float32)
    inference_seconds = time.perf_counter() - run_started
    return masked_lat.astype(np.float32), latents.astype(np.float32), load_seconds, inference_seconds


def torch_device(device_name: str):
    torch = require_module("torch")
    if device_name == "auto":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def run_torch(
    args: argparse.Namespace,
    device_name: str,
    masked_image: np.ndarray,
    mask64: np.ndarray,
    init_latents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, str]:
    torch = require_module("torch")
    require_module("diffusers")
    device = torch_device(device_name)

    sys.path.insert(0, str(args.pytorch_code_root))
    cwd = Path.cwd()
    os.chdir(args.pytorch_code_root)
    try:
        if "model_lib.nets.unet_gla" not in sys.modules:
            dummy_unet_gla = types.ModuleType("model_lib.nets.unet_gla")
            dummy_unet_gla.UNet2DGLAConditionModel = None
            sys.modules["model_lib.nets.unet_gla"] = dummy_unet_gla

        from diffusers import AutoencoderKL
        from removal.v1_2.removal_model import build_removal_model, load_cfg, load_removal_model

        load_started = time.perf_counter()
        cfg = load_cfg(str(args.pytorch_code_root / "config" / "model_cfg" / "moebius.yaml"))
        cfg["vae"]["model_dir"] = str(args.vae_dir)
        removal_model = build_removal_model(cfg, 20).eval()
        load_removal_model(
            removal_model,
            str(args.pytorch_weights_root / args.family / "diffusion_pytorch_model.bin"),
            "cpu",
        )
        removal_model = removal_model.to(device)
        vae = AutoencoderKL.from_pretrained(str(args.vae_dir)).eval().to(device)
        load_seconds = time.perf_counter() - load_started

        alphas_cumprod, timesteps = fixture.make_ddim(args.steps)
        timesteps = timesteps[max(args.steps - int(args.steps * args.strength), 0) :]
        latents = torch.tensor(init_latents, device=device)
        input_ids = torch.tensor(np.stack([np.arange(10, 20), np.arange(0, 10)]), dtype=torch.int64, device=device)
        mask_tensor = torch.tensor(mask64, device=device)

        run_started = time.perf_counter()
        with torch.no_grad():
            masked_lat = vae.encode(torch.tensor(masked_image, device=device)).latent_dist.mean * fixture.SCALING_FACTOR
            for index, timestep in enumerate(timesteps):
                previous = timesteps[index + 1] if index + 1 < len(timesteps) else -1
                nine = torch.cat([latents, mask_tensor, masked_lat], dim=1)
                nine2 = torch.cat([nine, nine], dim=0)
                ts = torch.tensor([int(timestep), int(timestep)], dtype=torch.int64, device=device)
                pred = removal_model(nine2, ts, input_ids).sample
                uncond, cond = pred[0:1], pred[1:2]
                cfg_pred = uncond + args.guidance * (cond - uncond)
                next_latents = fixture.ddim_step(
                    cfg_pred.detach().cpu().numpy(),
                    int(timestep),
                    latents.detach().cpu().numpy(),
                    alphas_cumprod,
                    int(previous),
                ).astype(np.float32)
                latents = torch.tensor(next_latents, device=device)
        inference_seconds = time.perf_counter() - run_started
        return (
            masked_lat.detach().cpu().numpy().astype(np.float32),
            latents.detach().cpu().numpy().astype(np.float32),
            load_seconds,
            inference_seconds,
            str(device),
        )
    finally:
        os.chdir(cwd)


def write_reference_fixture(args: argparse.Namespace, masked_lat: np.ndarray, mask64: np.ndarray, init_latents: np.ndarray, final_latents: np.ndarray) -> None:
    fixture_args = argparse.Namespace(
        output=args.output_dir / "reference_fixture",
        model=args.family,
        steps=args.steps,
        guidance=args.guidance,
        strength=args.strength,
        seed=args.seed,
    )
    fixture.write_fixture(fixture_args, masked_lat, mask64, init_latents, final_latents, args.reference_backend)


def parse_key_values(output: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


def build_swift_cli() -> Path:
    run_command(["swift", "build", "--product", "moebius-mlx"], cwd=ROOT)
    return ROOT / ".build" / "debug" / "moebius-mlx"


def download_mlx_variant(cli: Path, variant: str, download_base: Path, hf_token: str | None) -> Path:
    command = [str(cli), "download", variant, "--download-base", str(download_base)]
    if hf_token:
        command += ["--hf-token", hf_token]
    output = run_command(command, cwd=ROOT, capture=True)
    for line in output.splitlines():
        if line.startswith("directory:"):
            return Path(line.split(":", 1)[1].strip())
    raise RuntimeError(f"Unable to parse downloaded MLX directory for {variant}:\n{output}")


def resolve_mlx_variant(args: argparse.Namespace, cli: Path, variant: str) -> Path | str:
    if args.mlx_model_root:
        candidate = args.mlx_model_root / variant
        if candidate.exists():
            return candidate
    repo_candidate = ROOT / "Models" / "Moebius-MLX" / variant
    if repo_candidate.exists():
        return repo_candidate
    if not args.download_missing:
        return variant
    return download_mlx_variant(cli, variant, args.mlx_download_base, args.hf_token)


def run_mlx_variants(args: argparse.Namespace, reference: np.ndarray, reference_name: str) -> list[RuntimeRow]:
    cli = build_swift_cli()
    rows: list[RuntimeRow] = []
    for variant in args.mlx_variants:
        model_argument = resolve_mlx_variant(args, cli, variant)
        latent_path = args.output_dir / "latents" / f"{variant}.bin"
        output = run_command(
            [
                str(cli),
                "benchmark-fixture",
                str(model_argument),
                "--fixture",
                str(args.output_dir / "reference_fixture"),
                "--reference-name",
                args.reference_backend,
                "--output-latents",
                str(latent_path),
            ],
            cwd=ROOT,
            capture=True,
        )
        parsed = parse_key_values(output)
        rows.append(
            RuntimeRow(
                name=variant,
                backend="mlx",
                variant=variant,
                reference=reference_name,
                model_mb=float(parsed["model_mb_total"]),
                max_abs=float(parsed["max_abs"]),
                mean_abs=float(parsed["mean_abs"]),
                rmse=float(parsed["rmse"]),
                p99_abs=float(parsed["p99_abs"]),
                psnr_db=float(parsed["psnr_db"]),
                cosine=float(parsed["cosine_similarity"]),
                inference_seconds=float(parsed["inference_seconds"]),
                load_seconds=float(parsed["load_seconds"]),
                total_seconds=float(parsed["total_seconds"]),
                device="MLX",
            )
        )
        read_raw_tensor(latent_path, (1, 64, 64, 4))
    return rows


def row_from_values(
    name: str,
    backend: str,
    variant: str,
    reference_name: str,
    values: np.ndarray,
    reference: np.ndarray,
    model_mb: float,
    load_seconds: float,
    inference_seconds: float,
    device: str,
    notes: str = "",
) -> RuntimeRow:
    metrics = tensor_metrics(values, reference)
    return RuntimeRow(
        name=name,
        backend=backend,
        variant=variant,
        reference=reference_name,
        model_mb=model_mb,
        max_abs=metrics["max_abs"],
        mean_abs=metrics["mean_abs"],
        rmse=metrics["rmse"],
        p99_abs=metrics["p99_abs"],
        psnr_db=metrics["psnr_db"],
        cosine=metrics["cosine"],
        inference_seconds=inference_seconds,
        load_seconds=load_seconds,
        total_seconds=load_seconds + inference_seconds,
        device=device,
        notes=notes,
    )


def write_csv(path: Path, rows: list[RuntimeRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(RuntimeRow.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def finite(value: float) -> str:
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return "-"
    return f"{value:.6g}"


def import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_reports(args: argparse.Namespace, rows: list[RuntimeRow]) -> None:
    plt = import_matplotlib()
    names = [row.name for row in rows]

    def save_grouped(keys: list[str], title: str, ylabel: str, filename: str, log: bool = False) -> None:
        x = np.arange(len(rows))
        width = 0.8 / len(keys)
        fig, ax = plt.subplots(figsize=(13, 6))
        for index, key in enumerate(keys):
            ax.bar(x + (index - (len(keys) - 1) / 2) * width, [getattr(row, key) for row in rows], width=width, label=key)
        if log:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25, which="both")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output_dir / filename, dpi=160)
        plt.close(fig)

    save_grouped(["max_abs", "mean_abs", "rmse", "p99_abs"], "Fidelity vs Reference", "absolute latent error, log scale", "fidelity_errors.png", log=True)
    save_grouped(["inference_seconds", "load_seconds", "total_seconds"], "Runtime Performance", "seconds", "performance_timings.png")

    fig, ax = plt.subplots(figsize=(12, 5.8))
    bars = ax.bar(names, [row.model_mb for row in rows], color="#2563eb")
    ax.axhline(200, color="#dc2626", linestyle="--", linewidth=1.5)
    ax.text(len(rows) - 0.5, 200, " 200 MB target", color="#dc2626", va="bottom")
    ax.set_title("Model Package Size")
    ax.set_ylabel("MB")
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    for bar, row in zip(bars, rows):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{row.model_mb:.0f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "model_size.png", dpi=160)
    plt.close(fig)

    for x_key, filename, xlabel, title in [
        ("inference_seconds", "speed_fidelity_tradeoff.png", "inference seconds", "Speed vs Fidelity"),
        ("model_mb", "size_fidelity_tradeoff.png", "model package MB", "Size vs Fidelity"),
    ]:
        fig, ax = plt.subplots(figsize=(8.5, 6.2))
        for row in rows:
            ax.scatter(getattr(row, x_key), row.rmse, s=max(45, min(320, row.model_mb / 3)), alpha=0.75)
            ax.annotate(row.name.replace("ft_places2-", ""), (getattr(row, x_key), row.rmse), xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("RMSE vs reference")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(args.output_dir / filename, dpi=160)
        plt.close(fig)


def normalize_channel(channel: np.ndarray, symmetric: bool = False) -> Image.Image:
    data = channel.astype(np.float32)
    if symmetric:
        max_abs = max(float(np.max(np.abs(data))), 1e-8)
        normalized = (data / max_abs + 1.0) * 0.5
    else:
        low = float(np.quantile(data, 0.01))
        high = float(np.quantile(data, 0.99))
        if high - low < 1e-8:
            normalized = np.zeros_like(data)
        else:
            normalized = (data - low) / (high - low)
    return Image.fromarray(np.uint8(np.clip(normalized, 0, 1) * 255), mode="L").convert("RGB")


def make_latent_contact_sheet(args: argparse.Namespace, rows: list[RuntimeRow], reference: np.ndarray) -> None:
    latent_dir = args.output_dir / "latents"
    variants = [row.name for row in rows]
    tile = 96
    gap = 12
    label_w = 190
    top = 70
    row_h = tile * 2 + 48
    width = label_w + (tile + gap) * 4 + 40
    height = top + row_h * len(rows) + 20
    sheet = Image.new("RGB", (width, height), (246, 247, 249))
    draw = ImageDraw.Draw(sheet)
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)
        body_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 12)
    except OSError:
        title_font = body_font = ImageFont.load_default()
    draw.text((16, 18), "Moebius Cross-Runtime Latent Comparison", font=title_font, fill=(25, 29, 35))
    draw.text((16, 45), "Top row per variant: latent channels. Bottom row: absolute error vs the first PyTorch row.", font=body_font, fill=(73, 80, 87))

    ref_nhwc = reference.transpose(0, 2, 3, 1)[0]
    for row_index, name in enumerate(variants):
        row = rows[row_index]
        y = top + row_index * row_h
        path = latent_dir / f"{safe_name(name)}.bin"
        if not path.exists():
            continue
        values = read_raw_tensor(path, reference.shape if row.backend in {"onnx", "pytorch"} else (1, 64, 64, 4))
        nhwc = values.transpose(0, 2, 3, 1)[0] if values.shape[1] == 4 else values[0]
        diff = np.abs(nhwc - ref_nhwc)
        draw.text((16, y + 4), row.name, font=body_font, fill=(25, 29, 35))
        draw.text((16, y + 22), f"{row.backend}, {row.inference_seconds:.2f}s, RMSE {row.rmse:.4g}", font=body_font, fill=(73, 80, 87))
        for channel in range(4):
            x = label_w + channel * (tile + gap)
            latent_image = normalize_channel(nhwc[:, :, channel], symmetric=True).resize((tile, tile), Image.Resampling.NEAREST)
            diff_image = normalize_channel(diff[:, :, channel]).resize((tile, tile), Image.Resampling.NEAREST)
            sheet.paste(latent_image, (x, y))
            sheet.paste(diff_image, (x, y + tile + 8))
            draw.rectangle((x, y, x + tile, y + tile), outline=(207, 213, 219))
            draw.rectangle((x, y + tile + 8, x + tile, y + tile * 2 + 8), outline=(207, 213, 219))
            draw.text((x + 2, y - 14), f"ch {channel}", font=body_font, fill=(73, 80, 87))
    sheet.save(args.output_dir / "latent_contact_sheet.png")


def write_summary(args: argparse.Namespace, rows: list[RuntimeRow]) -> None:
    lines = [
        f"# Moebius {args.family} Cross-Runtime Benchmark",
        "",
        f"Reference: first requested PyTorch row. Steps: `{args.steps}`. Seed: `{args.seed}`. Guidance: `{args.guidance}`. Strength: `{args.strength}`.",
        "",
        "This benchmark compares deterministic latent denoising across PyTorch, ONNX, and MLX. CPU and accelerated PyTorch/ONNX rows are reported separately so performance comparisons can be made between matching device classes. The first requested PyTorch row is used only as the numeric fidelity reference because it is the original implementation; performance comparisons should be read row-by-row by backend/device.",
        "",
        "## Results",
        "",
        "| name | backend | model MB | max abs | mean abs | RMSE | p99 abs | PSNR dB | cosine | inference s | load s | total s | device |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.name} | {row.backend} | {row.model_mb:.2f} | {finite(row.max_abs)} | {finite(row.mean_abs)} | {finite(row.rmse)} | {finite(row.p99_abs)} | {finite(row.psnr_db)} | {row.cosine:.8f} | {row.inference_seconds:.3f} | {row.load_seconds:.3f} | {row.total_seconds:.3f} | {row.device} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Use `PyTorch MPS` versus `MLX` rows for the Apple GPU-backed speed comparison.",
            "- `PyTorch CPU` and `ONNX CPU` are included as explicit CPU baselines, not as the headline speed comparison.",
            "- `ONNX CoreML` uses ONNX Runtime's CoreML provider where available. Some graph nodes can still run on CPU, and provider setup is included in `load s`.",
            "- Fidelity metrics are measured against the first requested PyTorch row, which is the original implementation path used as a numeric reference.",
            "",
            "## Charts",
            "",
            "![Fidelity error metrics](fidelity_errors.png)",
            "",
            "![Runtime performance](performance_timings.png)",
            "",
            "![Model package size](model_size.png)",
            "",
            "![Speed vs fidelity](speed_fidelity_tradeoff.png)",
            "",
            "![Size vs fidelity](size_fidelity_tradeoff.png)",
            "",
            "## Contact Sheet",
            "",
            "![Latent contact sheet](latent_contact_sheet.png)",
            "",
            "## Files",
            "",
            "- `benchmark.csv`: raw machine-readable benchmark rows.",
            "- `reference_fixture/`: deterministic fixture used for MLX comparison.",
            "- `latents/`: raw f32 latent outputs used for metrics and the contact sheet.",
            "",
            "## Reproduce",
            "",
            "Install benchmark dependencies:",
            "",
            "```bash",
            "python3 -m venv .venv",
            "./.venv/bin/python -m pip install -r requirements-benchmarks.txt",
            "```",
            "",
            "Run the benchmark with local paths, or omit them to let the script download public artifacts:",
            "",
            "```bash",
            "./.venv/bin/python Scripts/run_cross_runtime_benchmark.py \\",
            "  --output-dir Docs/BenchmarkReports/ft_places2-cross-runtime-steps20",
            "```",
            "",
            "Useful path overrides:",
            "",
            "- `--pytorch-code-root`: original Moebius Python repository.",
            "- `--pytorch-weights-root`: Hugging Face `hustvl/Moebius` checkpoint root.",
            "- `--vae-dir`: PixelHacker VAE directory containing `config.json` and `diffusion_pytorch_model.bin`.",
            "- `--torch-devices`: comma-separated PyTorch devices, for example `cpu,mps`.",
            "- `--onnx-providers`: comma-separated ONNX Runtime providers, for example `CPUExecutionProvider,CoreMLExecutionProvider`.",
            "- `--onnx-root`: ONNX directory containing `vae_encoder.onnx` and `unet.onnx`.",
            "- `--mlx-model-root`: directory containing MLX variant folders.",
            "",
        ]
    )
    (args.output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="ft_places2", choices=["pretrained", "ft_places2", "ft_celebahq", "ft_ffhq"])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--strength", type=float, default=0.99)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--reference-backend", choices=["pytorch"], default="pytorch")
    parser.add_argument("--torch-devices", default="cpu,mps", help="Comma-separated PyTorch devices. Default: cpu,mps.")
    parser.add_argument("--torch-device", choices=["cpu", "mps", "auto"], help="Deprecated alias for --torch-devices with one device.")
    parser.add_argument("--onnx-providers", default="CPUExecutionProvider,CoreMLExecutionProvider", help="Comma-separated ONNX Runtime providers.")
    parser.add_argument("--onnx-provider", help="Deprecated alias for --onnx-providers with one provider.")
    parser.add_argument("--mlx-variants", default=",".join(MLX_VARIANTS))
    parser.add_argument("--mlx-model-root", type=Path, default=None)
    parser.add_argument("--mlx-download-base", type=Path)
    parser.add_argument("--pytorch-code-root", type=Path)
    parser.add_argument("--pytorch-weights-root", type=Path)
    parser.add_argument("--vae-dir", type=Path)
    parser.add_argument("--onnx-root", type=Path)
    parser.add_argument("--pytorch-code-repo", default="https://github.com/hustvl/Moebius.git")
    parser.add_argument("--pytorch-weights-repo", default="hustvl/Moebius")
    parser.add_argument("--vae-repo", default="hustvl/PixelHacker")
    parser.add_argument("--onnx-repo", default="simonw/Moebius-ONNX")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--no-download", action="store_true", help="Fail instead of downloading missing public model artifacts.")
    args = parser.parse_args()
    args.download_missing = not args.no_download
    args.torch_devices = parse_list(args.torch_device or args.torch_devices)
    args.onnx_providers = parse_list(args.onnx_provider or args.onnx_providers)
    args.mlx_variants = parse_list(args.mlx_variants)
    args.pytorch_code_root = args.pytorch_code_root or Path(os.environ.get("MOEBIUS_PYTORCH_CODE_ROOT", args.artifact_dir / "code" / "Moebius"))
    args.pytorch_weights_root = args.pytorch_weights_root or Path(os.environ.get("MOEBIUS_PYTORCH_WEIGHTS_ROOT", args.artifact_dir / "models" / "Moebius"))
    args.vae_dir = args.vae_dir or Path(os.environ.get("MOEBIUS_VAE_DIR", args.artifact_dir / "models" / "PixelHacker" / "vae"))
    args.onnx_root = args.onnx_root or Path(os.environ.get("MOEBIUS_ONNX_ROOT", args.artifact_dir / "models" / "Moebius-ONNX"))
    args.mlx_model_root = args.mlx_model_root or (Path(os.environ["MOEBIUS_MLX_MODEL_DIR"]) if os.environ.get("MOEBIUS_MLX_MODEL_DIR") else None)
    args.mlx_download_base = args.mlx_download_base or args.artifact_dir / "models" / "Moebius-MLX-Hub"
    code_root = args.pytorch_code_root
    args.image = args.image or code_root / "data" / "images" / "0.png"
    args.mask = args.mask or code_root / "data" / "masks" / "000000.png"
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "latents").mkdir(parents=True, exist_ok=True)

    ensure_reference_artifacts(args)
    _image_nchw, masked_image, mask64, init_latents = make_reference_inputs(args)

    torch_devices = available_torch_devices(args.torch_devices)
    if not torch_devices:
        raise SystemExit("No requested PyTorch devices are available.")

    rows: list[RuntimeRow] = []
    reference: np.ndarray | None = None
    reference_masked: np.ndarray | None = None
    reference_name = ""

    for index, device_name in enumerate(torch_devices):
        name = f"PyTorch {device_name.upper()}"
        print(f"Running {name}", file=sys.stderr)
        pytorch_masked, pytorch_final, pytorch_load, pytorch_inference, pytorch_device = run_torch(args, device_name, masked_image, mask64, init_latents)
        write_raw_tensor(args.output_dir / "latents" / f"{safe_name(name)}.bin", pytorch_final)
        if reference is None:
            reference = pytorch_final
            reference_masked = pytorch_masked
            reference_name = name
            write_reference_fixture(args, pytorch_masked, mask64, init_latents, pytorch_final)
        rows.append(
            row_from_values(
                name,
                "pytorch",
                args.family,
                reference_name,
                pytorch_final,
                reference,
                directory_size_mb(args.pytorch_weights_root / args.family) + directory_size_mb(args.vae_dir),
                pytorch_load,
                pytorch_inference,
                pytorch_device,
                notes="reference" if index == 0 else "",
            )
        )

    assert reference is not None
    assert reference_masked is not None

    onnx_providers = available_onnx_providers(args.onnx_providers)
    if not onnx_providers:
        print("No requested ONNX providers are available; skipping ONNX rows.", file=sys.stderr)
    for provider in onnx_providers:
        provider_label = provider.removesuffix("ExecutionProvider")
        name = f"ONNX {provider_label}"
        print(f"Running {name}", file=sys.stderr)
        _onnx_masked, onnx_final, onnx_load, onnx_inference = run_onnx(args, provider, masked_image, mask64, init_latents)
        write_raw_tensor(args.output_dir / "latents" / f"{safe_name(name)}.bin", onnx_final)
        rows.append(
            row_from_values(
                name,
                "onnx",
                args.family,
                reference_name,
                onnx_final,
                reference,
                directory_size_mb(args.onnx_root, [args.onnx_root / "vae_encoder.onnx", args.onnx_root / "unet.onnx", args.onnx_root / "vae_decoder.onnx"]),
                onnx_load,
                onnx_inference,
                provider,
            )
        )

    print("Running MLX variants", file=sys.stderr)
    rows.extend(run_mlx_variants(args, reference, reference_name))
    write_csv(args.output_dir / "benchmark.csv", rows)
    plot_reports(args, rows)
    make_latent_contact_sheet(args, rows, reference)
    write_summary(args, rows)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "family": args.family,
                "steps": args.steps,
                "seed": args.seed,
                "guidance": args.guidance,
                "strength": args.strength,
                "torch_devices": torch_devices,
                "onnx_providers": onnx_providers,
                "reference": reference_name,
                "mlx_variants": args.mlx_variants,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"report: {args.output_dir}")


if __name__ == "__main__":
    main()
