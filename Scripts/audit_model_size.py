#!/usr/bin/env python3
"""Audit Moebius artifact sizes by file, dtype, and rough tensor family."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from safetensors import safe_open


BYTE_UNITS = ["B", "KiB", "MiB", "GiB"]


@dataclass
class TensorRecord:
    file: str
    key: str
    dtype: str
    shape: tuple[int, ...]
    bytes: int
    elements: int
    family: str


def human_bytes(value: float) -> str:
    unit = 0
    while abs(value) >= 1024 and unit < len(BYTE_UNITS) - 1:
        value /= 1024
        unit += 1
    return f"{value:.2f} {BYTE_UNITS[unit]}"


def convolution_kernel(shape: tuple[int, ...]) -> str | None:
    if len(shape) != 4:
        return None
    if shape[1] <= 31 and shape[2] <= 31:
        return f"{shape[1]}x{shape[2]}"
    if shape[2] <= 31 and shape[3] <= 31:
        return f"{shape[2]}x{shape[3]}"
    return "other"


def tensor_family(file: str, key: str, shape: tuple[int, ...]) -> str:
    normalized = key.lower()
    if file.startswith("vae"):
        if ".conv" in normalized or normalized.endswith("conv_in.weight") or normalized.endswith("conv_out.weight"):
            kernel = convolution_kernel(shape)
            if kernel == "1x1":
                return "vae_conv_1x1"
            if kernel == "3x3":
                return "vae_conv_3x3"
            return "vae_conv_other"
        if any(part in normalized for part in ["to_q", "to_k", "to_v", "to_out", "query_proj", "key_proj", "value_proj", "out_proj", "quant_proj"]):
            return "vae_linear_attention"
        if ".norm" in normalized or "group_norm" in normalized:
            return "vae_norm"
        return "vae_other"

    if normalized.endswith(".scales") or normalized.endswith(".biases"):
        return "quant_metadata"
    if "embedding" in normalized:
        return "unet_embedding"
    if "time_emb_proj" in normalized or "time_embedding" in normalized or "encoder_hid_proj" in normalized:
        return "unet_linear_conditioning"
    if "ff.net" in normalized or "feed_forward" in normalized:
        return "unet_ffn"
    if "rel_pos_emb" in normalized:
        return "unet_relative_position"
    if ".attn2.to_k" in normalized or ".attn2.to_v" in normalized:
        return "unet_cross_linear"
    if ".conv" in normalized or ".to_q" in normalized or ".to_k" in normalized or ".to_v" in normalized or ".proj_" in normalized or ".downsample" in normalized or ".upsample" in normalized:
        kernel = convolution_kernel(shape)
        if kernel == "1x1":
            return "unet_conv_projection_1x1"
        if kernel == "3x3":
            return "unet_conv_projection_3x3"
        if kernel == "15x15":
            return "unet_conv_projection_15x15"
        return "unet_conv_projection_other"
    if ".norm" in normalized or ".bn" in normalized:
        return "unet_norm"
    return "unet_other"


def tensor_records(path: Path) -> list[TensorRecord]:
    records: list[TensorRecord] = []
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            elements = int(tensor.numel())
            bytes_count = int(elements * tensor.element_size())
            records.append(
                TensorRecord(
                    file=path.name,
                    key=key,
                    dtype=str(tensor.dtype).replace("torch.", ""),
                    shape=tuple(int(dim) for dim in tensor.shape),
                    bytes=bytes_count,
                    elements=elements,
                    family=tensor_family(path.name, key, tuple(int(dim) for dim in tensor.shape)),
                )
            )
    return records


def manifest_files(model_dir: Path, deployment: bool) -> list[Path]:
    manifest = json.loads((model_dir / "manifest.json").read_text())
    files = manifest["files"]
    vae_encoder = files.get("vaeEncoderQuantized") if deployment and manifest.get("quantizedVAEConvolutions") else None
    vae_decoder = files.get("vaeDecoderQuantized") if deployment and manifest.get("quantizedVAEConvolutions") else None
    names: list[str] = [
        "manifest.json",
        vae_encoder or files["vaeEncoder"],
        vae_decoder or files["vaeDecoder"],
    ]
    quantized = files.get("unetQuantized")
    if deployment and quantized:
        names.append(quantized)
    else:
        names.append(files["unet"])
        if quantized:
            names.append(quantized)
    return [model_dir / name for name in names]


def print_group(title: str, rows: Iterable[tuple[str, int, int]]) -> None:
    print(f"\n{title}")
    print("name\tbytes\tMiB\telements")
    for name, bytes_count, elements in sorted(rows, key=lambda row: row[1], reverse=True):
        print(f"{name}\t{bytes_count}\t{bytes_count / (1024 * 1024):.2f}\t{elements}")


def estimated_payload(records: list[TensorRecord], bits: int) -> int:
    total_bits = sum(record.elements for record in records) * bits
    return (total_bits + 7) // 8


def projected_payload(records: list[TensorRecord], families: set[str], bits: int) -> int:
    total = 0
    for record in records:
        if record.family in families:
            total += (record.elements * bits + 7) // 8
        else:
            total += record.bytes
    return total


def audit_model(model_dir: Path, deployment: bool, top: int, show_details: bool) -> tuple[str, int, int, dict[str, int], dict[str, int]]:
    files = manifest_files(model_dir, deployment=deployment)
    records: list[TensorRecord] = []
    manifest_bytes = 0
    for path in files:
        if path.name == "manifest.json":
            manifest_bytes += path.stat().st_size
            continue
        records.extend(tensor_records(path))

    total_bytes = sum(record.bytes for record in records) + manifest_bytes
    total_elements = sum(record.elements for record in records)

    by_dtype: dict[str, list[TensorRecord]] = defaultdict(list)
    by_family: dict[str, list[TensorRecord]] = defaultdict(list)
    by_file: dict[str, list[TensorRecord]] = defaultdict(list)
    for record in records:
        by_file[record.file].append(record)
        by_dtype[record.dtype].append(record)
        by_family[record.family].append(record)

    dtype_bytes: dict[str, int] = {name: sum(r.bytes for r in group) for name, group in by_dtype.items()}
    family_bytes: dict[str, int] = {name: sum(r.bytes for r in group) for name, group in by_family.items()}

    if show_details:
        print(f"model_dir: {model_dir}")
        print(f"mode: {'deployment' if deployment else 'conversion-directory'}")
        print(f"files: {', '.join(path.name for path in files)}")
        print(f"total_bytes_payload: {total_bytes}")
        print(f"total_size: {human_bytes(total_bytes)}")
        print(f"total_elements: {total_elements}")
        print(f"ideal_q8_payload_no_metadata: {human_bytes(estimated_payload(records, 8) + manifest_bytes)}")
        print(f"ideal_q4_payload_no_metadata: {human_bytes(estimated_payload(records, 4) + manifest_bytes)}")

        projection_groups = {
            "q4_unet_1x1_conv_only": {"unet_conv_projection_1x1"},
            "q4_unet_all_conv": {
                "unet_conv_projection_1x1",
                "unet_conv_projection_3x3",
                "unet_conv_projection_15x15",
                "unet_conv_projection_other",
            },
            "q4_unet_conv_and_relpos": {
                "unet_conv_projection_1x1",
                "unet_conv_projection_3x3",
                "unet_conv_projection_15x15",
                "unet_conv_projection_other",
                "unet_relative_position",
            },
            "q4_unet_and_vae_conv": {
                "unet_conv_projection_1x1",
                "unet_conv_projection_3x3",
                "unet_conv_projection_15x15",
                "unet_conv_projection_other",
                "unet_relative_position",
                "vae_conv_1x1",
                "vae_conv_3x3",
                "vae_conv_other",
            },
        }
        print("\nProjected payloads, no extra metadata")
        print("scenario\tbytes\tMiB")
        for name, families in projection_groups.items():
            projected = projected_payload(records, families, 4) + manifest_bytes
            print(f"{name}\t{projected}\t{projected / (1024 * 1024):.2f}")

        print_group("By file", ((name, sum(r.bytes for r in group), sum(r.elements for r in group)) for name, group in by_file.items()))
        print_group("By dtype", ((name, sum(r.bytes for r in group), sum(r.elements for r in group)) for name, group in by_dtype.items()))
        print_group("By family", ((name, sum(r.bytes for r in group), sum(r.elements for r in group)) for name, group in by_family.items()))

        print(f"\nTop {top} tensors")
        print("file\tfamily\tdtype\tMiB\tshape\tkey")
        for record in sorted(records, key=lambda item: item.bytes, reverse=True)[:top]:
            print(
                f"{record.file}\t{record.family}\t{record.dtype}\t{record.bytes / (1024 * 1024):.2f}\t"
                f"{'x'.join(str(dim) for dim in record.shape)}\t{record.key}"
            )

    return model_dir.name, total_bytes, total_elements, dtype_bytes, family_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path, nargs="+")
    parser.add_argument("--deployment", action="store_true", help="Count only runtime files, using unetQuantized when present.")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--details", action="store_true", help="Print full per-model breakdowns when auditing multiple models.")
    args = parser.parse_args()

    show_details = len(args.model_dir) == 1 or args.details
    summaries = [
        audit_model(model_dir, deployment=args.deployment, top=args.top, show_details=show_details)
        for model_dir in args.model_dir
    ]

    if len(summaries) > 1:
        print("model\tMiB\tMB\telements\tfloat16 MiB\tuint32 MiB\tother MiB")
        for name, total_bytes, total_elements, dtype_bytes, _ in summaries:
            float16 = dtype_bytes.get("float16", 0) / (1024 * 1024)
            uint32 = dtype_bytes.get("uint32", 0) / (1024 * 1024)
            other = (sum(dtype_bytes.values()) - dtype_bytes.get("float16", 0) - dtype_bytes.get("uint32", 0)) / (1024 * 1024)
            print(
                f"{name}\t{total_bytes / (1024 * 1024):.2f}\t{total_bytes / 1_000_000:.2f}\t"
                f"{total_elements}\t{float16:.2f}\t{uint32:.2f}\t{other:.2f}"
            )


if __name__ == "__main__":
    main()
