"""
YOLO26n 训练与导出脚本。

功能：
1. 自动识别数据集目录
2. 兼容 labels/train 和 labels/tarin
3. 训练前强制执行数据审计，拒绝连续帧泄漏和缺失类别
4. 训练 YOLO 检测模型
5. 训练完成后自动导出 best.pt 为 NCNN 格式
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent

import torch
import ultralytics
from ultralytics import YOLO


DATASET_ROOT_CANDIDATES = [
    ROOT / "dataset" / "robocup",
    ROOT / "datasets" / "robocup",
]
DATA_YAML = ROOT / "robocup_data.yaml"
PRETRAINED_WEIGHTS = "yolo26n.pt"
PROJECT_DIR = ROOT / "runs" / "detect"
RUN_NAME = "train_clean"
IMG_SIZE = 320
EPOCHS = 150
DEFAULT_BATCH = 16
SEED = 42
BEST_WEIGHTS = PROJECT_DIR / RUN_NAME / "weights" / "best.pt"
EXPECTED_ULTRALYTICS_VERSION = "8.4.21"


def resolve_dataset_root() -> Path:
    """找到实际存在的数据集根目录。"""
    for candidate in DATASET_ROOT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "未找到数据集根目录，请确认存在 dataset/robocup 或 datasets/robocup。"
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


def audit_dataset(dataset_root: Path) -> None:
    """在训练前检查标签、类别、会话隔离和划分清单是否仍然有效。"""
    command = [
        sys.executable,
        "-X",
        "utf8",
        str(ROOT / "dataset_audit.py"),
        "--dataset",
        str(dataset_root),
    ]
    subprocess.run(command, check=True)


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
        "translate": 0.15,
        "scale": 0.5,
        "perspective": 0.001,
        "hsv_h": 0.01,
        "hsv_s": 0.5,
        "hsv_v": 0.4,
        # 类别本身依赖红/紫与黄绿/蓝图案，不使用不真实的 BGR 通道互换。
        "bgr": 0.0,
        "project": str(PROJECT_DIR),
        "name": RUN_NAME,
        "exist_ok": True,
        "close_mosaic": 10,
        "seed": SEED,
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
    if ultralytics.__version__ != EXPECTED_ULTRALYTICS_VERSION:
        warnings.warn(
            f"建议使用 ultralytics=={EXPECTED_ULTRALYTICS_VERSION}，"
            f"当前为 {ultralytics.__version__}，指标与导出后处理可能不同。",
            RuntimeWarning,
        )
    dataset_root = resolve_dataset_root()
    train_labels_dir = ensure_train_labels(dataset_root)
    audit_dataset(dataset_root)

    class_ids = detect_class_ids(train_labels_dir)
    if class_ids != {0, 1}:
        raise RuntimeError(
            f"期望训练类别为 0/1（增益/减益），实际检测到：{sorted(class_ids)}"
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
