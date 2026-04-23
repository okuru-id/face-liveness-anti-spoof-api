#!/usr/bin/env python3
import argparse
import csv
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.vendor.mini_fasnet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE
from app.vendor.silent_face_utility import get_kernel, parse_model_name


MODEL_MAPPING = {
    "MiniFASNetV1": MiniFASNetV1,
    "MiniFASNetV2": MiniFASNetV2,
    "MiniFASNetV1SE": MiniFASNetV1SE,
    "MiniFASNetV2SE": MiniFASNetV2SE,
}


@dataclass
class Sample:
    image_path: Path
    split: str
    is_live: bool


class AntiSpoofDataset(Dataset):
    def __init__(self, samples: list[Sample], train: bool):
        self.original_samples = list(samples)
        self.train = train
        self.samples: list[Sample] = list(samples)

    def oversample_minority(self, target_ratio: float = 0.33) -> None:
        if not self.train:
            return
        live = [s for s in self.samples if s.is_live]
        spoof = [s for s in self.samples if not s.is_live]
        if not live or not spoof:
            return
        target_live = max(int(len(spoof) * target_ratio), len(live))
        if target_live <= len(live):
            return
        multiplier = target_live // len(live)
        remainder = target_live % len(live)
        oversampled = live * multiplier + live[:remainder]
        self.samples = spoof + oversampled
        random.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = cv2.imread(str(sample.image_path))
        if image is None:
            raise RuntimeError(f"Gagal baca image: {sample.image_path}")

        image = cv2.resize(image, (80, 80), interpolation=cv2.INTER_LINEAR)
        if self.train:
            image = self._augment(image)

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        x = torch.from_numpy(image)

        # Tetap kompatibel dengan inference existing:
        # label 1 = LIVE, label selain 1 dianggap SPOOF.
        y = 1 if sample.is_live else 0
        return x, torch.tensor(y, dtype=torch.long)

    def _augment(self, image: np.ndarray) -> np.ndarray:
        out = image

        if random.random() < 0.5:
            out = cv2.flip(out, 1)

        if random.random() < 0.7:
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-15.0, 15.0)
            out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

        if random.random() < 0.25:
            k = random.choice([3, 5])
            out = cv2.GaussianBlur(out, (k, k), 0)

        return out


def _load_manifest(manifest_path: Path, data_root: Path) -> list[Sample]:
    rows: list[Sample] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel = row["image_path"].strip()
            split = row["split"].strip()
            label = row["label"].strip().lower()
            rows.append(
                Sample(
                    image_path=(data_root / rel).resolve(),
                    split=split,
                    is_live=(label == "live"),
                )
            )
    return rows


def _split_samples(samples: list[Sample]):
    train = [s for s in samples if s.split == "train"]
    val = [s for s in samples if s.split == "val"]
    test = [s for s in samples if s.split == "test"]
    return train, val, test


def _load_pretrained(model: nn.Module, checkpoint_path: Path) -> None:
    if not checkpoint_path.exists():
        return
    state_dict = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    fixed = {}
    for k, v in state_dict.items():
        fixed[k[7:] if k.startswith("module.") else k] = v

    model_state = model.state_dict()
    for k in model_state.keys():
        if k not in fixed or fixed[k].shape != model_state[k].shape:
            fixed[k] = model_state[k]

    model.load_state_dict(fixed, strict=False)


def _compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    preds = torch.argmax(logits, dim=1)
    pred_live = preds == 1
    true_live = targets == 1

    total = float(targets.numel())
    acc = float((preds == targets).sum().item() / max(total, 1.0))

    tp = float((pred_live & true_live).sum().item())
    tn = float((~pred_live & ~true_live).sum().item())
    fp = float((pred_live & ~true_live).sum().item())
    fn = float((~pred_live & true_live).sum().item())

    apcer = fp / max(fp + tn, 1.0)  # spoof yang salah jadi live
    bpcer = fn / max(fn + tp, 1.0)  # live yang salah jadi spoof
    acer = (apcer + bpcer) / 2.0

    return {
        "acc": acc,
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": acer,
    }


def _evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, float]:
    model.eval()
    losses = []
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            losses.append(float(loss.item()))
            all_logits.append(logits.cpu())
            all_targets.append(y.cpu())

    if not all_logits:
        return {"loss": math.inf, "acc": 0.0, "apcer": 1.0, "bpcer": 1.0}

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = _compute_metrics(logits, targets)
    metrics["loss"] = float(sum(losses) / max(len(losses), 1))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune MiniFASNet dengan manifest dataset custom.")
    parser.add_argument(
        "--manifest",
        default=str(PROJECT_ROOT / "training_data" / "antispoof" / "manifest.csv"),
        help="Path manifest.csv dari script prepare_antispoof_dataset.py",
    )
    parser.add_argument(
        "--data-root",
        default=str(PROJECT_ROOT / "training_data" / "antispoof"),
        help="Root folder dataset hasil prepare",
    )
    parser.add_argument(
        "--model-name",
        default="2.7_80x80_MiniFASNetV2.pth",
        choices=[
            "2.7_80x80_MiniFASNetV2.pth",
            "4_0_0_80x80_MiniFASNetV1SE.pth",
        ],
        help="Nama model basis agar kompatibel dengan inference existing",
    )
    parser.add_argument(
        "--init-weights",
        default="",
        help="Path checkpoint awal (.pth). Kosong = pakai models/<model-name> jika ada.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "models" / "finetuned"),
        help="Folder output checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-class-weights", action="store_true", help="Matikan class-weighted loss")
    return parser.parse_args()


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest_path = Path(args.manifest)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = _load_manifest(manifest_path, data_root)
    train_samples, val_samples, test_samples = _split_samples(samples)
    if not train_samples:
        raise RuntimeError("Split train kosong. Periksa manifest/split.")

    train_ds = AntiSpoofDataset(train_samples, train=True)
    train_ds.oversample_minority(target_ratio=0.33)
    val_ds = AntiSpoofDataset(val_samples, train=False)
    test_ds = AntiSpoofDataset(test_samples, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    h_input, w_input, model_type, _ = parse_model_name(args.model_name)
    kernel = get_kernel(h_input, w_input)
    model = MODEL_MAPPING[model_type](conv6_kernel=kernel).to(device)

    init_weights = Path(args.init_weights) if args.init_weights else (PROJECT_ROOT / "models" / args.model_name)
    _load_pretrained(model, init_weights)

    criterion: nn.Module
    if args.no_class_weights:
        criterion = nn.CrossEntropyLoss()
    else:
        train_live = sum(1 for s in train_ds.samples if s.is_live)
        train_spoof = sum(1 for s in train_ds.samples if not s.is_live)
        w_spoof = (train_live + train_spoof) / max(train_spoof, 1)
        w_live = (train_live + train_spoof) / max(train_live, 1)
        class_weights = torch.tensor([w_spoof, w_live, 0.5], dtype=torch.float32, device=device)
        criterion = FocalLoss(gamma=2.0, weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_val_acer = float("inf")
    best_val_loss = float("inf")
    best_path = output_dir / f"best_{args.model_name}"

    print(f"device={device}")
    print(f"train={len(train_ds)} (live={sum(1 for s in train_ds.samples if s.is_live)}, spoof={sum(1 for s in train_ds.samples if not s.is_live)}) val={len(val_ds)} test={len(test_ds)}")
    print(f"init_weights={init_weights}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        scheduler.step()

        train_loss = float(sum(train_losses) / max(len(train_losses), 1))
        val_metrics = _evaluate(model, val_loader, criterion, device) if len(val_ds) > 0 else {
            "loss": train_loss,
            "acc": 0.0,
            "apcer": 1.0,
            "bpcer": 1.0,
            "acer": 1.0,
        }

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"val_apcer={val_metrics['apcer']:.4f} "
            f"val_bpcer={val_metrics['bpcer']:.4f} "
            f"val_acer={val_metrics['acer']:.4f}"
        )

        improved = (val_metrics["acer"] < best_val_acer) or (
            math.isclose(val_metrics["acer"], best_val_acer) and val_metrics["loss"] < best_val_loss
        )
        if improved:
            best_val_acer = val_metrics["acer"]
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), best_path)

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    test_metrics = _evaluate(model, test_loader, criterion, device) if len(test_ds) > 0 else {
        "loss": 0.0,
        "acc": 0.0,
        "apcer": 0.0,
        "bpcer": 0.0,
        "acer": 0.0,
    }

    print(
        "best_model="
        f"{best_path} "
        f"test_loss={test_metrics['loss']:.4f} "
        f"test_acc={test_metrics['acc']:.4f} "
        f"test_apcer={test_metrics['apcer']:.4f} "
        f"test_bpcer={test_metrics['bpcer']:.4f} "
        f"test_acer={test_metrics['acer']:.4f}"
    )


if __name__ == "__main__":
    main()
