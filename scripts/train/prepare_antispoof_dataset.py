#!/usr/bin/env python3
import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.face_detector import face_detector


def _scaled_crop(image: np.ndarray, bbox: tuple[int, int, int, int], scale: float = 2.7) -> np.ndarray:
    x, y, box_w, box_h = bbox
    src_h, src_w = image.shape[:2]
    scale = min((src_h - 1) / max(box_h, 1), min((src_w - 1) / max(box_w, 1), scale))
    new_width = box_w * scale
    new_height = box_h * scale
    center_x, center_y = box_w / 2 + x, box_h / 2 + y
    left_top_x = center_x - new_width / 2
    left_top_y = center_y - new_height / 2
    right_bottom_x = center_x + new_width / 2
    right_bottom_y = center_y + new_height / 2

    if left_top_x < 0:
        right_bottom_x -= left_top_x
        left_top_x = 0
    if left_top_y < 0:
        right_bottom_y -= left_top_y
        left_top_y = 0
    if right_bottom_x > src_w - 1:
        left_top_x -= right_bottom_x - src_w + 1
        right_bottom_x = src_w - 1
    if right_bottom_y > src_h - 1:
        left_top_y -= right_bottom_y - src_h + 1
        right_bottom_y = src_h - 1

    left_top_x = int(max(left_top_x, 0))
    left_top_y = int(max(left_top_y, 0))
    right_bottom_x = int(min(right_bottom_x, src_w - 1))
    right_bottom_y = int(min(right_bottom_y, src_h - 1))
    return image[left_top_y : right_bottom_y + 1, left_top_x : right_bottom_x + 1]


def _sample_video_frames(video_path: Path, max_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[np.ndarray] = []

    if frame_count > 0:
        indices = sorted(set(np.linspace(0, frame_count - 1, num=max_frames, dtype=int).tolist()))
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
    else:
        step = 5
        idx = 0
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                frames.append(frame)
            idx += 1

    cap.release()
    return frames


def _split_groups(group_ids: list[str], train_ratio: float, val_ratio: float, rng: random.Random) -> dict[str, str]:
    ids = list(group_ids)
    rng.shuffle(ids)
    n = len(ids)
    if n == 0:
        return {}

    train_count = int(round(n * train_ratio))
    val_count = int(round(n * val_ratio))
    if train_count <= 0:
        train_count = 1
    if train_count >= n:
        train_count = n - 1
    if val_count < 0:
        val_count = 0
    if train_count + val_count >= n:
        val_count = max(0, n - train_count - 1)

    mapping: dict[str, str] = {}
    for i, gid in enumerate(ids):
        if i < train_count:
            mapping[gid] = "train"
        elif i < train_count + val_count:
            mapping[gid] = "val"
        else:
            mapping[gid] = "test"
    return mapping


HARD_NEGATIVE_FORCE_TRAIN = {"spoof-extra::spoof-9", "spoof-extra::spoof-10"}


def _assign_split(rows: list[dict], train_ratio: float, val_ratio: float, seed: int) -> None:
    rng = random.Random(seed)
    live_groups = sorted({r["group_id"] for r in rows if r["label"] == "live"})
    spoof_groups = sorted({r["group_id"] for r in rows if r["label"] == "spoof"})

    split_map = {}
    split_map.update(_split_groups(live_groups, train_ratio, val_ratio, rng))
    split_map.update(_split_groups(spoof_groups, train_ratio, val_ratio, rng))

    for gid in HARD_NEGATIVE_FORCE_TRAIN:
        split_map[gid] = "train"

    for row in rows:
        row["split"] = split_map.get(row["group_id"], "train")


def _save_crop(
    frame: np.ndarray,
    label: str,
    spoof_type: str,
    source_dataset: str,
    source_file: str,
    group_id: str,
    output_root: Path,
    counters: dict[str, int],
    rows: list[dict],
    jpeg_quality: int,
) -> bool:
    face = face_detector.detect(frame)
    if not face.detected or not face.bbox:
        return False

    crop = _scaled_crop(frame, face.bbox, scale=2.7)
    if crop.size == 0:
        return False

    crop = cv2.resize(crop, (80, 80), interpolation=cv2.INTER_LINEAR)
    label_dir = output_root / "images" / label
    label_dir.mkdir(parents=True, exist_ok=True)

    counters[label] = counters.get(label, 0) + 1
    file_name = f"{label}_{counters[label]:06d}.jpg"
    out_path = label_dir / file_name
    cv2.imwrite(str(out_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])

    rows.append(
        {
            "image_path": out_path.relative_to(output_root).as_posix(),
            "label": label,
            "label_id": 1 if label == "live" else 2,
            "spoof_type": spoof_type,
            "source_dataset": source_dataset,
            "source_file": source_file,
            "group_id": group_id,
        }
    )
    return True


def build_dataset(
    dataset_live_root: Path,
    dataset_attack_root: Path,
    extra_labeled_dir: Path | None,
    output_root: Path,
    max_frames_per_video: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    jpeg_quality: int,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    counters: dict[str, int] = {"live": 0, "spoof": 0}

    real_csv = dataset_live_root / "real_30.csv"
    samples_root = dataset_live_root / "samples"
    if real_csv.exists():
        with real_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                worker_id = (row.get("worker_id") or "unknown").strip() or "unknown"

                selfie_rel = (row.get("selfie_link") or "").strip().lstrip("/")
                if selfie_rel:
                    selfie_path = samples_root / selfie_rel
                    if selfie_path.exists():
                        image = cv2.imread(str(selfie_path))
                        if image is not None:
                            _save_crop(
                                frame=image,
                                label="live",
                                spoof_type="none",
                                source_dataset="anti-spoofing",
                                source_file=str(selfie_path),
                                group_id=f"live1::{worker_id}",
                                output_root=output_root,
                                counters=counters,
                                rows=rows,
                                jpeg_quality=jpeg_quality,
                            )

                video_rel = (row.get("video_link") or "").strip().lstrip("/")
                if video_rel:
                    video_path = samples_root / video_rel
                    if video_path.exists():
                        frames = _sample_video_frames(video_path, max_frames_per_video)
                        for frame in frames:
                            _save_crop(
                                frame=frame,
                                label="live",
                                spoof_type="none",
                                source_dataset="anti-spoofing",
                                source_file=str(video_path),
                                group_id=f"live1::{worker_id}",
                                output_root=output_root,
                                counters=counters,
                                rows=rows,
                                jpeg_quality=jpeg_quality,
                            )

    attacks_csv = dataset_attack_root / "webcam_attacks.csv"
    files_root = dataset_attack_root / "files"
    if attacks_csv.exists():
        with attacks_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel = (row.get("file") or "").strip().lstrip("/")
                attack_type = (row.get("type") or "unknown").strip().lower()
                if not rel:
                    continue

                path = files_root / rel
                if not path.exists():
                    continue

                label = "live" if attack_type == "real" else "spoof"
                spoof_type = "none" if label == "live" else attack_type
                group_id = f"live2::{Path(rel).stem}" if label == "live" else f"spoof2::{attack_type}::{Path(rel).stem}"

                frames = _sample_video_frames(path, max_frames_per_video)
                for frame in frames:
                    _save_crop(
                        frame=frame,
                        label=label,
                        spoof_type=spoof_type,
                        source_dataset="anti-spoofing-1",
                        source_file=str(path),
                        group_id=group_id,
                        output_root=output_root,
                        counters=counters,
                        rows=rows,
                        jpeg_quality=jpeg_quality,
                    )

    if extra_labeled_dir is not None and extra_labeled_dir.exists() and extra_labeled_dir.is_dir():
        for image_path in sorted(extra_labeled_dir.iterdir()):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            name = image_path.name.lower()
            if "live" in name:
                label = "live"
                spoof_type = "none"
                group_id = f"live-extra::{image_path.stem}"
            elif "spoof" in name:
                label = "spoof"
                spoof_type = "unknown"
                group_id = f"spoof-extra::{image_path.stem}"
            else:
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                continue

            _save_crop(
                frame=image,
                label=label,
                spoof_type=spoof_type,
                source_dataset="data-test",
                source_file=str(image_path),
                group_id=group_id,
                output_root=output_root,
                counters=counters,
                rows=rows,
                jpeg_quality=jpeg_quality,
            )

    if not rows:
        raise RuntimeError("Tidak ada sample berhasil diproses. Periksa path dataset dan model face detector.")

    _assign_split(rows, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)

    fieldnames = [
        "image_path",
        "label",
        "label_id",
        "spoof_type",
        "source_dataset",
        "source_file",
        "group_id",
        "split",
    ]

    manifest_all = output_root / "manifest.csv"
    with manifest_all.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    split_counts = {"train": 0, "val": 0, "test": 0}
    label_counts = {"live": 0, "spoof": 0}
    for row in rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
        label_counts[row["label"]] = label_counts.get(row["label"], 0) + 1

    print(f"Manifest: {manifest_all}")
    print(f"Total sample: {len(rows)}")
    print(f"Label count: {label_counts}")
    print(f"Split count: {split_counts}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Siapkan dataset training anti-spoof dari 2 sumber folder.")
    parser.add_argument(
        "--dataset-live-root",
        default="/home/kurob/Documents/KERJAAN/AI/dataset/anti-spoofing",
        help="Path root dataset live (punya real_30.csv dan folder samples/)",
    )
    parser.add_argument(
        "--dataset-attack-root",
        default="/home/kurob/Documents/KERJAAN/AI/dataset/anti-spoofing-1",
        help="Path root dataset attack (punya webcam_attacks.csv dan folder files/)",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "training_data" / "antispoof"),
        help="Folder output untuk face crop dan manifest",
    )
    parser.add_argument(
        "--extra-labeled-dir",
        default=str(PROJECT_ROOT / "data-test"),
        help="Folder image tambahan berlabel via nama file (mengandung 'live'/'spoof')",
    )
    parser.add_argument("--max-frames-per-video", type=int, default=8)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(
        dataset_live_root=Path(args.dataset_live_root),
        dataset_attack_root=Path(args.dataset_attack_root),
        extra_labeled_dir=Path(args.extra_labeled_dir) if args.extra_labeled_dir else None,
        output_root=Path(args.output_root),
        max_frames_per_video=max(1, args.max_frames_per_video),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        jpeg_quality=max(60, min(100, args.jpeg_quality)),
    )


if __name__ == "__main__":
    main()
