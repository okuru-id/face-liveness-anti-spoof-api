#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/export_pth_to_onnx.sh --pth <input.pth> [--onnx <output.onnx>] [options]

Options:
  --pth <path>         Path file .pth input (wajib)
  --onnx <path>        Path output .onnx (default: ganti ekstensi .pth -> .onnx)
  --type <auto|mini-fasnet|physnet>
                       Tipe model (default: auto)
  --opset <int>        ONNX opset (default: 13)
  --frames <int>       Jumlah frame untuk PhysNet (default: 6)
  --batch <int>        Dummy batch size (default: 1)
  --help               Tampilkan bantuan

Contoh:
  scripts/export_pth_to_onnx.sh \
    --pth models/active/antispoof/best_2.7_80x80_MiniFASNetV2.pth

  scripts/export_pth_to_onnx.sh \
    --type physnet \
    --frames 6 \
    --pth models/active/rppg/UBFC-rPPG_PhysNet_DiffNormalized.pth \
    --onnx models/active/rppg/UBFC-rPPG_PhysNet_DiffNormalized.onnx
EOF
}

PTH_PATH=""
ONNX_PATH=""
MODEL_TYPE="auto"
OPSET="13"
FRAMES="6"
BATCH="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pth)
      PTH_PATH="${2:-}"
      shift 2
      ;;
    --onnx)
      ONNX_PATH="${2:-}"
      shift 2
      ;;
    --type)
      MODEL_TYPE="${2:-}"
      shift 2
      ;;
    --opset)
      OPSET="${2:-}"
      shift 2
      ;;
    --frames)
      FRAMES="${2:-}"
      shift 2
      ;;
    --batch)
      BATCH="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Argumen tidak dikenal: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$PTH_PATH" ]]; then
  echo "Error: --pth wajib diisi" >&2
  usage
  exit 1
fi

if [[ -z "$ONNX_PATH" ]]; then
  ONNX_PATH="${PTH_PATH%.pth}.onnx"
fi

export PTH_PATH
export ONNX_PATH
export MODEL_TYPE
export OPSET
export FRAMES
export BATCH

python3 - <<'PY'
import math
import os
from pathlib import Path

import torch
import torch.nn as nn


def normalize_state_dict(raw_state):
    if isinstance(raw_state, dict) and "state_dict" in raw_state:
        raw_state = raw_state["state_dict"]
    if not isinstance(raw_state, dict):
        raise RuntimeError("Format checkpoint tidak didukung")

    fixed = {}
    for key, value in raw_state.items():
        fixed[key[7:] if key.startswith("module.") else key] = value
    return fixed


def detect_model_type(model_path: Path) -> str:
    name = model_path.name.lower()
    if "minifasnet" in name:
        return "mini-fasnet"
    if "physnet" in name or "rppg" in name:
        return "physnet"
    raise RuntimeError(
        "Gagal auto-detect tipe model. Gunakan --type mini-fasnet atau --type physnet."
    )


def export_minifasnet(pth_path: Path, onnx_path: Path, opset: int, batch: int):
    from app.vendor.mini_fasnet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE
    from app.vendor.silent_face_utility import get_kernel, parse_model_name

    model_mapping = {
        "MiniFASNetV1": MiniFASNetV1,
        "MiniFASNetV2": MiniFASNetV2,
        "MiniFASNetV1SE": MiniFASNetV1SE,
        "MiniFASNetV2SE": MiniFASNetV2SE,
    }

    h_input, w_input, model_type, _ = parse_model_name(pth_path.name)
    if model_type not in model_mapping:
        raise RuntimeError(f"Tipe MiniFASNet tidak didukung: {model_type}")

    kernel_size = get_kernel(h_input, w_input)
    model = model_mapping[model_type](conv6_kernel=kernel_size)
    model.load_state_dict(normalize_state_dict(torch.load(str(pth_path), map_location="cpu")), strict=False)
    model.eval()

    dummy = torch.randn(batch, 3, h_input, w_input, dtype=torch.float32)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )


def export_physnet(pth_path: Path, onnx_path: Path, opset: int, batch: int, frames: int):
    from app.vendor.physnet_model import PhysNet_padding_Encoder_Decoder_MAX

    model = PhysNet_padding_Encoder_Decoder_MAX(frames=frames)
    model.load_state_dict(normalize_state_dict(torch.load(str(pth_path), map_location="cpu")), strict=False)
    model.eval()

    class PhysNetWrapper(nn.Module):
        def __init__(self, base_model, target_frames: int):
            super().__init__()
            self.base_model = base_model
            self.target_frames = target_frames

        def _adaptive_temporal_pool_exact(self, x: torch.Tensor) -> torch.Tensor:
            source_len = int(x.shape[2])
            if source_len == self.target_frames:
                return x

            pooled_segments = []
            for i in range(self.target_frames):
                start = int(math.floor(i * source_len / self.target_frames))
                end = int(math.ceil((i + 1) * source_len / self.target_frames))
                segment = x[:, :, start:end, :, :]
                pooled_segments.append(segment.mean(dim=2, keepdim=True))
            return torch.cat(pooled_segments, dim=2)

        def forward(self, x):
            _, _, length, _, _ = x.shape

            x = self.base_model.ConvBlock1(x)
            x = self.base_model.MaxpoolSpa(x)

            x = self.base_model.ConvBlock2(x)
            x = self.base_model.ConvBlock3(x)
            x = self.base_model.MaxpoolSpaTem(x)

            x = self.base_model.ConvBlock4(x)
            x = self.base_model.ConvBlock5(x)
            x = self.base_model.MaxpoolSpaTem(x)

            x = self.base_model.ConvBlock6(x)
            x = self.base_model.ConvBlock7(x)
            x = self.base_model.MaxpoolSpa(x)

            x = self.base_model.ConvBlock8(x)
            x = self.base_model.ConvBlock9(x)
            x = self.base_model.upsample(x)
            x = self.base_model.upsample2(x)

            # Hindari AdaptiveAvgPool3d((frames,1,1)) saat export ONNX ketika
            # output temporal bukan faktor dari frames (contoh: 4 -> 6).
            x = x.mean(dim=(-2, -1), keepdim=True)
            x = self._adaptive_temporal_pool_exact(x)
            x = self.base_model.ConvBlock10(x)
            rppg = x.view(-1, length)
            return rppg

    wrapped = PhysNetWrapper(model, target_frames=frames).eval()
    dummy = torch.randn(batch, 3, frames, 128, 128, dtype=torch.float32)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapped,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["rppg"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "rppg": {0: "batch_size"},
        },
    )


pth_path = Path(os.environ["PTH_PATH"]).resolve()
onnx_path = Path(os.environ["ONNX_PATH"]).resolve()
model_type = os.environ.get("MODEL_TYPE", "auto").strip().lower()
opset = int(os.environ.get("OPSET", "13"))
frames = int(os.environ.get("FRAMES", "6"))
batch = int(os.environ.get("BATCH", "1"))

if not pth_path.exists():
    raise FileNotFoundError(f"File .pth tidak ditemukan: {pth_path}")

if model_type == "auto":
    model_type = detect_model_type(pth_path)

if model_type == "mini-fasnet":
    export_minifasnet(pth_path, onnx_path, opset, batch)
elif model_type == "physnet":
    export_physnet(pth_path, onnx_path, opset, batch, frames)
else:
    raise RuntimeError(f"--type tidak valid: {model_type}")

print(f"OK: {pth_path} -> {onnx_path} (type={model_type}, opset={opset})")
PY
