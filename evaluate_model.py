"""Evaluate a YOLO checkpoint on the session-isolated validation split."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在干净验证集上评估模型")
    parser.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "runs" / "detect" / "train" / "weights" / "best.pt",
    )
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default="clean_val")
    parser.add_argument("--plots", action="store_true", help="生成评估图表")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subprocess.run(
        [sys.executable, "-X", "utf8", str(ROOT / "dataset_audit.py")],
        cwd=ROOT,
        check=True,
    )
    if not args.weights.exists():
        raise FileNotFoundError(args.weights)

    device: int | str = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(ROOT / "robocup_data.yaml"),
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=0,
        plots=args.plots,
        project=str(ROOT / "runs" / "detect"),
        name=args.name,
        exist_ok=True,
    )

    print("\n按类别 mAP50-95：")
    evaluated_class_ids = [int(value) for value in metrics.box.ap_class_index]
    for class_id in evaluated_class_ids:
        class_name = model.names[class_id]
        print(f"- {class_id} {class_name}: {metrics.box.maps[class_id]:.4f}")
    print(f"总体 mAP50: {metrics.box.map50:.4f}")
    print(f"总体 mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
