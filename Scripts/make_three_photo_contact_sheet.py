#!/usr/bin/env python3
"""Build a three-photo Moebius visual comparison sheet and metrics tables."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


@dataclass(frozen=True)
class Variant:
    key: str
    title: str
    subtitle: str
    filename_key: str


PHOTOS: tuple[tuple[str, str], ...] = (
    ("0", "000000"),
    ("1", "000001"),
    ("100", "000003"),
)

VARIANTS: tuple[Variant, ...] = (
    Variant("pytorch", "PyTorch", "reference", "pytorch"),
    Variant("onnx", "ONNX", "ONNX CPU", "onnx"),
    Variant("ft_places2-f32", "MLX f32", "", "ft_places2-f32"),
    Variant("ft_places2-f16", "MLX f16", "", "ft_places2-f16"),
    Variant("ft_places2-mxfp8", "MLX mxfp8", "", "ft_places2-mxfp8"),
    Variant("ft_places2-mxfp4", "MLX mxfp4", "", "ft_places2-mxfp4"),
    Variant("ft_places2-q4", "MLX q4", "", "ft_places2-q4"),
    Variant("ft_places2-q4-ff-spatial-vae", "MLX q4 322", "ff+spatial+VAE", "ft_places2-q4-ff-spatial-vae"),
    Variant("ft_places2-q4-all-spatial-vae", "MLX q4 200", "all+spatial+VAE", "ft_places2-q4-all-spatial-vae"),
)


def load_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def to_image(values: np.ndarray) -> Image.Image:
    clipped = np.clip(values, 0.0, 1.0)
    return Image.fromarray(np.round(clipped * 255.0).astype(np.uint8), mode="RGB")


def edge_map(values: np.ndarray) -> np.ndarray:
    gray = np.dot(values[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
    image = Image.fromarray(np.round(np.clip(gray, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    return np.asarray(image.filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0


def abs_diff_image(reference: np.ndarray, candidate: np.ndarray, scale: float) -> Image.Image:
    return to_image(np.abs(candidate - reference) * scale)


def edge_diff_image(reference: np.ndarray, candidate: np.ndarray, scale: float) -> Image.Image:
    diff = np.abs(edge_map(candidate) - edge_map(reference))
    return Image.fromarray(np.round(np.clip(diff * scale, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L").convert("RGB")


def signed_luma_image(reference: np.ndarray, candidate: np.ndarray, scale: float) -> Image.Image:
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    delta = np.dot(candidate - reference, weights) * scale
    red = np.clip(delta, 0.0, 1.0)
    blue = np.clip(-delta, 0.0, 1.0)
    green = np.zeros_like(red)
    return to_image(np.stack([red, green, blue], axis=-1))


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    delta = (candidate - reference).reshape(-1)
    abs_delta = np.abs(delta)
    mse = float(np.mean(delta * delta))
    rmse = math.sqrt(mse)
    psnr = float("inf") if mse == 0 else 20.0 * math.log10(1.0 / math.sqrt(mse))
    ref_flat = reference.reshape(-1)
    cand_flat = candidate.reshape(-1)
    denom = float(np.linalg.norm(ref_flat) * np.linalg.norm(cand_flat))
    cosine = 1.0 if denom == 0 else float(np.dot(ref_flat, cand_flat) / denom)
    return {
        "mean_abs": float(np.mean(abs_delta)),
        "rmse": rmse,
        "max_abs": float(np.max(abs_delta)),
        "p99_abs": float(np.quantile(abs_delta, 0.99)),
        "psnr_db": psnr,
        "cosine": cosine,
    }


def average_metrics(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(rows)
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def model_size_mb(model_root: Path, variant_key: str) -> float | None:
    if variant_key in {"pytorch", "onnx"}:
        return None
    directory = model_root / variant_key
    if not directory.exists():
        return None
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    files = manifest["files"]
    use_quantized_vae = bool(manifest.get("quantizedVAEConvolutions")) and files.get("vaeEncoderQuantized") and files.get("vaeDecoderQuantized")
    selected_files = [
        "manifest.json",
        files.get("unetQuantized") or files["unet"],
        files["vaeEncoderQuantized"] if use_quantized_vae else files["vaeEncoder"],
        files["vaeDecoderQuantized"] if use_quantized_vae else files["vaeDecoder"],
    ]
    total = sum((directory / name).resolve().stat().st_size for name in selected_files)
    return total / 1_000_000.0


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: tuple[int, int, int]) -> None:
    draw.text(xy, text, font=font, fill=fill)


def build_sheet(args: argparse.Namespace) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    input_dir = Path(args.input_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model_root = Path(args.model_root)

    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)
        header_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 14)
        body_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 12)
        small_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 11)
    except OSError:
        title_font = header_font = body_font = small_font = ImageFont.load_default()

    loaded: dict[tuple[str, str], np.ndarray] = {}
    per_variant_metrics_vs_pytorch: dict[str, list[dict[str, float]]] = {variant.key: [] for variant in VARIANTS}
    for photo, _mask in PHOTOS:
        reference = load_image(input_dir / f"photo-{photo}_pytorch.png")
        for variant in VARIANTS:
            values = load_image(input_dir / f"photo-{photo}_{variant.filename_key}.png")
            loaded[(photo, variant.key)] = values
            per_variant_metrics_vs_pytorch[variant.key].append(metrics(reference, values))

    per_variant_metrics_vs_f32: dict[str, list[dict[str, float]]] = {variant.key: [] for variant in VARIANTS}
    for photo, _mask in PHOTOS:
        reference = loaded[(photo, "ft_places2-f32")]
        for variant in VARIANTS:
            per_variant_metrics_vs_f32[variant.key].append(metrics(reference, loaded[(photo, variant.key)]))

    avg_metrics_vs_pytorch = {key: average_metrics(rows) for key, rows in per_variant_metrics_vs_pytorch.items()}
    avg_metrics_vs_f32 = {key: average_metrics(rows) for key, rows in per_variant_metrics_vs_f32.items()}

    tile = 132
    gap_x = 14
    gap_y = 26
    label_w = 178
    top = 128
    left = 16
    header_h = 54
    row_h = (tile * 4) + (gap_y * 3) + 42
    width = left + label_w + (tile + gap_x) * len(VARIANTS) + 20
    height = top + row_h * len(PHOTOS) + 24
    sheet = Image.new("RGB", (width, height), (246, 247, 249))
    draw = ImageDraw.Draw(sheet)

    draw_text(draw, (16, 18), "Moebius ft_places2 three-photo PyTorch-reference comparison", title_font, (25, 29, 35))
    draw_text(draw, (16, 45), f"{args.steps} steps, seed {args.seed}. Diff rows compare each column to PyTorch for the same photo.", body_font, (73, 80, 87))

    for column, variant in enumerate(VARIANTS):
        x = left + label_w + column * (tile + gap_x)
        draw_text(draw, (x, 78), variant.title, header_font, (25, 29, 35))
        subtitle = variant.subtitle
        size_mb = model_size_mb(model_root, variant.key)
        if size_mb is not None:
            subtitle = f"{size_mb:.2f} MB" if not subtitle else f"{subtitle}, {size_mb:.2f} MB"
        draw_text(draw, (x, 99), subtitle, small_font, (73, 80, 87))

    for photo_index, (photo, mask) in enumerate(PHOTOS):
        y0 = top + photo_index * row_h
        reference = loaded[(photo, "pytorch")]
        labels = [
            (f"photo {photo} / mask {mask}", "output"),
            ("abs diff vs PyTorch x4", ""),
            ("edge-map diff vs PyTorch x8", ""),
            ("signed luminance heatmap x5", "red = brighter, blue = darker"),
        ]
        for row_index, (label, sublabel) in enumerate(labels):
            y = y0 + header_h + row_index * (tile + gap_y)
            draw_text(draw, (left, y - 20), label, body_font, (25, 29, 35))
            if sublabel:
                draw_text(draw, (left, y - 4), sublabel, small_font, (73, 80, 87))

        for column, variant in enumerate(VARIANTS):
            candidate = loaded[(photo, variant.key)]
            x = left + label_w + column * (tile + gap_x)
            output_y = y0 + header_h
            if variant.key == "pytorch":
                images = [
                    to_image(candidate),
                    Image.new("RGB", (256, 256), (0, 0, 0)),
                    Image.new("RGB", (256, 256), (0, 0, 0)),
                    Image.new("RGB", (256, 256), (0, 0, 0)),
                ]
            else:
                images = [
                    to_image(candidate),
                    abs_diff_image(reference, candidate, 4.0),
                    edge_diff_image(reference, candidate, 8.0),
                    signed_luma_image(reference, candidate, 5.0),
                ]
            for row_index, image in enumerate(images):
                y = output_y + row_index * (tile + gap_y)
                thumb = image.resize((tile, tile), Image.Resampling.LANCZOS)
                sheet.paste(thumb, (x, y))
                draw.rectangle((x, y, x + tile, y + tile), outline=(207, 213, 219), width=1)

    sheet.save(output)
    return avg_metrics_vs_pytorch, avg_metrics_vs_f32


def write_metric_files(metrics_by_variant: dict[str, dict[str, float]], args: argparse.Namespace, baseline: str) -> None:
    output = Path(args.output)
    json_path = output.with_name(f"quant_metrics_vs_{baseline}.json")
    tsv_path = output.with_name(f"quant_metrics_vs_{baseline}.tsv")
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(metrics_by_variant, file, indent=2, sort_keys=True)
        file.write("\n")
    fields = ["variant", "mean_abs", "rmse", "max_abs", "p99_abs", "psnr_db", "cosine"]
    with tsv_path.open("w", encoding="utf-8") as file:
        file.write("\t".join(fields) + "\n")
        for variant in VARIANTS:
            row = metrics_by_variant[variant.key]
            values = [variant.key] + [f"{row[field]:.9g}" for field in fields[1:]]
            file.write("\t".join(values) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-root", default="/Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX")
    parser.add_argument("--steps", default="20")
    parser.add_argument("--seed", default="7")
    args = parser.parse_args()
    metrics_vs_pytorch, metrics_vs_f32 = build_sheet(args)
    write_metric_files(metrics_vs_pytorch, args, baseline="pytorch")
    write_metric_files(metrics_vs_f32, args, baseline="mlx_f32")
    print(args.output)


if __name__ == "__main__":
    main()
