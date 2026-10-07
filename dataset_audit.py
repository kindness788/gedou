"""Audit YOLO labels and create session-isolated train/validation manifests.

The source images are never moved or deleted. A capture date is treated as one
session so adjacent video frames cannot leak across train and validation.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
EXPECTED_CLASS_IDS = {0, 1}
SESSION_RE = re.compile(r"^(?:WIN|frame)_(\d{8})_")


@dataclass(frozen=True)
class Sample:
    image: Path
    label: Path
    session: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="审计 YOLO 数据，并按拍摄日期生成无连续帧泄漏的划分清单"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset" / "robocup",
    )
    parser.add_argument(
        "--write-splits",
        action="store_true",
        help="生成 splits/train.txt 和 splits/val.txt，不移动原始文件",
    )
    parser.add_argument(
        "--val-session",
        help="作为验证集的拍摄日期，例如 20260816；默认使用最新日期",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="允许覆盖已有划分清单",
    )
    return parser.parse_args()


def session_from_name(stem: str) -> str:
    match = SESSION_RE.match(stem)
    return match.group(1) if match else "unknown"


def collect_samples(dataset_root: Path) -> list[Sample]:
    image_dir = dataset_root / "images" / "train"
    label_dir = dataset_root / "labels" / "train"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError("需要 dataset/robocup/images/train 和 labels/train")

    samples = []
    for image in sorted(image_dir.iterdir()):
        if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES:
            samples.append(
                Sample(
                    image=image.resolve(),
                    label=(label_dir / f"{image.stem}.txt").resolve(),
                    session=session_from_name(image.stem),
                )
            )
    if not samples:
        raise RuntimeError("训练图片目录为空")
    return samples


def read_labels(sample: Sample) -> tuple[Counter[int], list[str], bool]:
    errors: list[str] = []
    counts: Counter[int] = Counter()
    if not sample.label.exists():
        return counts, errors, True

    lines = [line.strip() for line in sample.label.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]
    for line_number, line in enumerate(lines, 1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{sample.label.name}:{line_number}: 字段数不是 5")
            continue
        try:
            class_id = int(parts[0])
            x, y, width, height = map(float, parts[1:])
        except ValueError:
            errors.append(f"{sample.label.name}:{line_number}: 标签无法解析")
            continue

        counts[class_id] += 1
        if class_id not in EXPECTED_CLASS_IDS:
            errors.append(f"{sample.label.name}:{line_number}: 未配置的类别 {class_id}")
        values = (x, y, width, height)
        if any(value < 0.0 or value > 1.0 for value in values):
            errors.append(f"{sample.label.name}:{line_number}: 坐标不在 0..1")
        if (
            x - width / 2 < -1e-6
            or x + width / 2 > 1 + 1e-6
            or y - height / 2 < -1e-6
            or y + height / 2 > 1 + 1e-6
        ):
            errors.append(f"{sample.label.name}:{line_number}: 边界框越过图像范围")
    return counts, errors, not lines


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(path: Path, samples: list[Sample], force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} 已存在；确认后使用 --force 覆盖")
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(sample.image.as_posix() for sample in samples) + "\n"
    path.write_text(content, encoding="utf-8")


def manifest_paths(path: Path) -> set[Path]:
    if not path.exists():
        return set()
    return {
        Path(line.strip()).resolve()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def audit_split(name: str, samples: list[Sample]) -> tuple[list[str], Counter[int]]:
    errors: list[str] = []
    classes: Counter[int] = Counter()
    negative_images = 0
    for sample in samples:
        sample_counts, sample_errors, is_negative = read_labels(sample)
        classes.update(sample_counts)
        errors.extend(sample_errors)
        negative_images += int(is_negative)

    missing_classes = EXPECTED_CLASS_IDS - set(classes)
    if missing_classes:
        errors.append(f"{name} 缺少类别：{sorted(missing_classes)}")

    sessions = sorted({sample.session for sample in samples})
    print(
        f"{name}: images={len(samples)}, instances={dict(classes)}, "
        f"negative_images={negative_images}, sessions={sessions}"
    )
    if negative_images == 0:
        print(f"警告：{name} 没有纯背景负样本，建议补充真实赛场背景。")
    return errors, classes


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset.resolve()
    samples = collect_samples(dataset_root)
    sessions = sorted({sample.session for sample in samples})
    if "unknown" in sessions:
        raise RuntimeError("存在无法从文件名识别拍摄日期的图片，请先统一命名")
    if len(sessions) < 2:
        raise RuntimeError("至少需要两个独立拍摄日期才能进行分组验证")

    val_session = args.val_session or sessions[-1]
    if val_session not in sessions:
        raise ValueError(f"验证会话 {val_session} 不存在；可选值：{sessions}")

    train_samples = [sample for sample in samples if sample.session != val_session]
    val_samples = [sample for sample in samples if sample.session == val_session]

    errors = []
    split_sessions = {
        "train": {sample.session for sample in train_samples},
        "val": {sample.session for sample in val_samples},
    }
    overlap = split_sessions["train"] & split_sessions["val"]
    if overlap:
        errors.append(f"train/val 拍摄会话重叠：{sorted(overlap)}")

    split_hashes = {}
    for name, split_samples in (("train", train_samples), ("val", val_samples)):
        split_errors, _ = audit_split(name, split_samples)
        errors.extend(split_errors)
        split_hashes[name] = {sha256(sample.image) for sample in split_samples}
    duplicates = split_hashes["train"] & split_hashes["val"]
    if duplicates:
        errors.append(f"train/val 有 {len(duplicates)} 张字节级重复图片")

    if not args.write_splits:
        split_dir = dataset_root / "splits"
        expected = {
            "train": {sample.image for sample in train_samples},
            "val": {sample.image for sample in val_samples},
        }
        for name in ("train", "val"):
            manifest = split_dir / f"{name}.txt"
            actual = manifest_paths(manifest)
            if not actual:
                errors.append(f"缺少或为空：{manifest}；请先运行 --write-splits")
            elif actual != expected[name]:
                errors.append(
                    f"{manifest} 已过期（缺少 {len(expected[name] - actual)}，"
                    f"多出 {len(actual - expected[name])}）；请用 --write-splits --force 更新"
                )

    if errors:
        print("\n审计失败：")
        for error in errors[:50]:
            print(f"- {error}")
        if len(errors) > 50:
            print(f"- 其余 {len(errors) - 50} 项已省略")
        return 2

    if args.write_splits:
        split_dir = dataset_root / "splits"
        write_manifest(split_dir / "train.txt", train_samples, args.force)
        write_manifest(split_dir / "val.txt", val_samples, args.force)
        print(f"已写入：{split_dir / 'train.txt'}")
        print(f"已写入：{split_dir / 'val.txt'}")

    print("\n审计通过。测试集仍应使用未来新采集、从未参与调参的独立场次。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
