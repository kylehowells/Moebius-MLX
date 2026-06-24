#!/usr/bin/env python3
"""Convert Moebius PyTorch artifacts into the Moebius-MLX model layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


ROOT = Path("/Users/kylehowells/Developer/ML-Models/Moebius")
DEFAULT_WEIGHTS = ROOT / "Moebius-Models"
DEFAULT_OUT = ROOT / "Moebius-MLX"
DEFAULT_VAE = ROOT / "PixelHacker" / "vae" / "diffusion_pytorch_model.bin"


def load_torch_state(path: Path) -> dict[str, torch.Tensor]:
	state = torch.load(path, map_location="cpu")
	if isinstance(state, dict) and "state_dict" in state:
		state = state["state_dict"]
	if not isinstance(state, dict):
		raise TypeError(f"Unsupported checkpoint format: {path}")
	out: dict[str, torch.Tensor] = {}
	for key, value in state.items():
		if isinstance(value, torch.Tensor):
			out[key] = value.detach().cpu()
	return out


def convert_dtype(tensor: torch.Tensor, precision: str) -> torch.Tensor:
	if not torch.is_floating_point(tensor):
		return tensor
	if precision != "f32":
		return tensor.to(torch.float16)
	return tensor.to(torch.float32)


def save_safetensors(tensors: dict[str, torch.Tensor], path: Path, precision: str) -> None:
	try:
		from safetensors.torch import save_file
	except ImportError as exc:
		raise SystemExit("Install safetensors: pip install safetensors") from exc

	path.parent.mkdir(parents=True, exist_ok=True)
	converted = {key: convert_dtype(value, precision).contiguous() for key, value in tensors.items()}
	save_file(converted, str(path))


def write_manifest(
	out_dir: Path,
	model_name: str,
	checkpoint: Path,
	precision: str,
	unet_file: str,
	vae_encoder_file: str,
	vae_decoder_file: str,
) -> None:
	manifest: dict[str, Any] = {
		"formatVersion": 1,
		"modelName": model_name,
		"sourceCheckpoint": str(checkpoint),
		"precision": precision,
		"imageSize": 512,
		"latentSize": 64,
		"latentChannels": 4,
		"maskChannels": 1,
		"conditioningIDs": 20,
		"scalingFactor": 0.13025,
		"noiseOffset": 0.0357,
		"scheduler": {
			"betaStart": 0.00085,
			"betaEnd": 0.012,
			"trainTimesteps": 1000,
			"betaSchedule": "scaled_linear",
			"clipSample": False,
		},
		"files": {
			"unet": unet_file,
			"vaeEncoder": vae_encoder_file,
			"vaeDecoder": vae_decoder_file,
		},
	}
	(out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--weights-root", type=Path, default=DEFAULT_WEIGHTS)
	parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
	parser.add_argument("--vae", type=Path, default=DEFAULT_VAE)
	parser.add_argument("--model", choices=["pretrained", "ft_places2", "ft_celebahq", "ft_ffhq"], default="ft_places2")
	parser.add_argument("--precision", choices=["f32", "f16", "q8", "q4", "mxfp8", "mxfp4"], default="f32")
	args = parser.parse_args()

	checkpoint = args.weights_root / args.model / "diffusion_pytorch_model.bin"
	if not checkpoint.exists():
		raise SystemExit(f"Missing checkpoint: {checkpoint}")

	variant_name = f"{args.model}-{args.precision}"
	out_dir = args.out_root / variant_name
	out_dir.mkdir(parents=True, exist_ok=True)

	print(f"Loading UNet checkpoint: {checkpoint}")
	unet = load_torch_state(checkpoint)
	save_safetensors(unet, out_dir / "unet.safetensors", args.precision)

	if not args.vae.exists():
		raise SystemExit(f"Missing VAE checkpoint: {args.vae}")
	print(f"Loading VAE checkpoint: {args.vae}")
	vae = load_torch_state(args.vae)
	vae_encoder = {
		key: value
		for key, value in vae.items()
		if key.startswith("encoder.") or key.startswith("quant_conv.")
	}
	vae_decoder = {
		key: value
		for key, value in vae.items()
		if key.startswith("decoder.") or key.startswith("post_quant_conv.")
	}
	save_safetensors(vae_encoder, out_dir / "vae_encoder.safetensors", args.precision)
	save_safetensors(vae_decoder, out_dir / "vae_decoder.safetensors", args.precision)

	write_manifest(
		out_dir=out_dir,
		model_name=args.model,
		checkpoint=checkpoint,
		precision=args.precision,
		unet_file="unet.safetensors",
		vae_encoder_file="vae_encoder.safetensors",
		vae_decoder_file="vae_decoder.safetensors",
	)
	print(f"Wrote {out_dir}")


if __name__ == "__main__":
	main()
