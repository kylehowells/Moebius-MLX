#!/usr/bin/env python3
"""Create stable model-directory views for experimental quantized candidates."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def relink(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        target.unlink()
    os.symlink(source, target)


def materialize(root: Path, name: str, source_name: str, updates: dict) -> Path:
    source = root / source_name
    target = root / name
    if not source.is_dir():
        raise SystemExit(f"Missing source variant: {source}")
    target.mkdir(parents=True, exist_ok=True)

    for child in source.iterdir():
        target_child = target / child.name
        if child.name == "manifest.json":
            continue
        if target_child.exists() or target_child.is_symlink():
            if target_child.is_dir() and not target_child.is_symlink():
                shutil.rmtree(target_child)
            else:
                target_child.unlink()
        relink(child, target_child)

    manifest = json.loads((source / "manifest.json").read_text())
    files = manifest.setdefault("files", {})
    for key, value in updates.get("files", {}).items():
        files[key] = value
    for key, value in updates.items():
        if key == "files":
            continue
        if value is None:
            manifest.pop(key, None)
        else:
            manifest[key] = value
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX"))
    args = parser.parse_args()

    candidates = {
        "ft_places2-q4-ff-spatial-vae": {
            "source": "ft_places2-q4",
            "updates": {
                "files": {
                    "unetQuantized": "unet_quantized_pointwise_feedforward_spatial.safetensors",
                    "vaeEncoderQuantized": "vae_encoder_quantized.safetensors",
                    "vaeDecoderQuantized": "vae_decoder_quantized.safetensors",
                },
                "quantizedPointwiseConvolutions": True,
                "quantizedPointwiseConvolutionPolicy": "feedforward",
                "quantizedSpatialConvolutions": True,
                "quantizedSpatialConvolutionPolicy": "spatial",
                "quantizedVAEConvolutions": True,
                "quantizedVAEConvolutionPolicy": "vae-conv",
            },
        },
        "ft_places2-q4-all-spatial-vae": {
            "source": "ft_places2-q4",
            "updates": {
                "files": {
                    "unetQuantized": "unet_quantized_pointwise_all_spatial.safetensors",
                    "vaeEncoderQuantized": "vae_encoder_quantized.safetensors",
                    "vaeDecoderQuantized": "vae_decoder_quantized.safetensors",
                },
                "quantizedPointwiseConvolutions": True,
                "quantizedPointwiseConvolutionPolicy": "all",
                "quantizedSpatialConvolutions": True,
                "quantizedSpatialConvolutionPolicy": "spatial",
                "quantizedVAEConvolutions": True,
                "quantizedVAEConvolutionPolicy": "vae-conv",
            },
        },
    }

    for name, spec in candidates.items():
        target = materialize(args.root, name, spec["source"], spec["updates"])
        print(target)


if __name__ == "__main__":
    main()
