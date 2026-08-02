"""
YOLO26n 训练与导出脚本。

功能：
1. 自动识别数据集目录
2. 兼容 labels/train 和 labels/tarin
3. 如果没有 val，则自动从 train 中切分一部分作为验证集
4. 训练 YOLO 检测模型
5. 训练完成后自动导出 best.pt 为 NCNN 格式
"""

from __future__ import annotations

import random
import shutil
import warnings
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATASET_ROOT_CANDIDATES = [
    ROOT / "dataset" / "robocup",
    ROOT / "datasets" / "robocup",
]
DATA_YAML = ROOT / "robocup_data.yaml"
PRETRAINED_WEIGHTS = "yolo26n.pt"
PROJECT_DIR = ROOT / "runs" / "detect"
RUN_NAME = "train"
IMG_SIZE = 320
EPOCHS = 150
DEFAULT_BATCH = 16
VAL_RATIO = 0.1
SEED = 42
BEST_WEIGHTS = PROJECT_DIR / RUN_NAME / "weights" / "best.pt"


def resolve_dataset_root() -> Path:
    """找到实际存在的数据集根目录。"""
    for candidate in DATASET_ROOT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "未找到数据集根目录，请确认存在 dataset/robocup 或 datasets/robocup。"
    )


def list_images(image_dir: Path) -> list[Path]:
    """列出图片文件。"""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )


def copy_label_tree(src_dir: Path, dst_dir: Path) -> None:
    """把标签目录复制到标准结构下。"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for file_path in src_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() == ".txt":
            shutil.copy2(file_path, dst_dir / file_path.name)


def ensure_train_labels(dataset_root: Path) -> Path:
    """
    保证训练标签目录存在，并返回标准化后的标签目录。

    你的当前目录里是 labels/tarin，所以这里会自动复制成 labels/train。
    """
    labels_root = dataset_root / "labels"
    standard_train = labels_root / "train"
    typo_train = labels_root / "tarin"

    if standard_train.exists():
        return standard_train

    if typo_train.exists():
        copy_label_tree(typo_train, standard_train)
        return standard_train

    raise FileNotFoundError(f"未找到标签目录：{standard_train} 或 {typo_train}")


def split_val_set(dataset_root: Path, train_labels_dir: Path) -> None:
    """
    如果没有 val，则从训练集中自动切出一份验证集。
    这里使用复制，不修改原始数据。
    """
    images_train = dataset_root / "images" / "train"
    images_val = dataset_root / "images" / "val"
    labels_val = dataset_root / "labels" / "val"

    if images_val.exists() and labels_val.exists():
        return

    images = list_images(images_train)
    if not images:
        raise FileNotFoundError(f"训练图片目录为空：{images_train}")

    paired = []
    for img_path in images:
        label_path = train_labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            paired.append((img_path, label_path))
        else:
            warnings.warn(f"跳过缺失标签的图片：{img_path.name}", RuntimeWarning)

    if len(paired) < 2:
        warnings.warn("可用于切分的样本太少，暂不自动创建 val。", RuntimeWarning)
        return

    random.seed(SEED)
    random.shuffle(paired)

    val_count = max(1, int(len(paired) * VAL_RATIO))
    val_pairs = paired[:val_count]

    images_val.mkdir(parents=True, exist_ok=True)
    labels_val.mkdir(parents=True, exist_ok=True)

    for img_path, label_path in val_pairs:
        shutil.copy2(img_path, images_val / img_path.name)
        shutil.copy2(label_path, labels_val / label_path.name)


def detect_class_ids(labels_dir: Path) -> set[int]:
    """统计当前标签中实际出现的类别 id。"""
    class_ids: set[int] = set()
    for txt_path in labels_dir.glob("*.txt"):
        with txt_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    class_ids.add(int(parts[0]))
                except ValueError:
                    continue
    return class_ids


def build_train_kwargs() -> dict:
    """构建训练参数。"""
    return {
        "data": str(DATA_YAML),
        "epochs": EPOCHS,
        "imgsz": IMG_SIZE,
        "batch": DEFAULT_BATCH,
        "mosaic": 0.5,
        "degrees": 15,
        "bgr": 0.2,
        "project": str(PROJECT_DIR),
        "name": RUN_NAME,
        "exist_ok": True,
        "close_mosaic": 10,
        "device": 0 if torch.cuda.is_available() else "cpu",
    }


def find_best_weights() -> Path:
    """查找训练生成的 best.pt。"""
    candidates = [
        BEST_WEIGHTS,
        ROOT / "runs" / "detect" / "runs" / "detect" / RUN_NAME / "weights" / "best.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(
        (p for p in (ROOT / "runs").rglob("best.pt")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if matches:
        return matches[0]

    raise FileNotFoundError("未找到 best.pt，请先确认训练是否完成。")


def train_model() -> YOLO:
    """训练模型，显存不足时自动降为 batch=-1。"""
    model = YOLO(PRETRAINED_WEIGHTS)
    train_kwargs = build_train_kwargs()

    try:
        model.train(**train_kwargs)
    except Exception as exc:
        message = str(exc).lower()
        if "out of memory" in message or "cuda" in message:
            warnings.warn("检测到显存压力，自动改为 batch=-1 自适应训练。", RuntimeWarning)
            train_kwargs["batch"] = -1
            model.train(**train_kwargs)
        else:
            raise

    return model


def export_best_to_ncnn() -> Path:
    """读取 best.pt 并导出为 NCNN。"""
    best_weights = find_best_weights()
    best_model = YOLO(str(best_weights))
    exported = best_model.export(
        format="ncnn",
        imgsz=IMG_SIZE,
    )

    if isinstance(exported, (list, tuple)):
        exported = exported[0]
    if exported is None:
        exported = best_weights.parent / "best_ncnn_model"

    return Path(exported)


def main() -> None:
    """主流程：准备数据 -> 训练 -> 导出。"""
    dataset_root = resolve_dataset_root()
    train_labels_dir = ensure_train_labels(dataset_root)
    split_val_set(dataset_root, train_labels_dir)

    class_ids = detect_class_ids(train_labels_dir)
    if class_ids == {0}:
        warnings.warn(
            "当前标签里只检测到类别 0。若你要训练 3 类，请确认 1、2 类也已标注。",
            RuntimeWarning,
        )

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"未找到数据集配置文件：{DATA_YAML}")

    print("开始训练 YOLO26n 检测模型...")
    train_model()

    best_weights = find_best_weights()
    print(f"开始导出最优权重：{best_weights}")
    ncnn_path = export_best_to_ncnn()
    print(f"导出完成：{ncnn_path}")


if __name__ == "__main__":
    main()
