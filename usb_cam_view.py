"""
USB 摄像头实时推理脚本。

用途：
1. 直接加载训练好的 .pt 权重
2. 调用 USB 摄像头实时显示检测结果
3. 适合快速查看模型在真实场景下的运行效果

默认参数：
1. 权重：runs/detect/train/weights/best.pt
2. 摄像头：0
3. 输入尺寸：320
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"
DEFAULT_CAMERA = 0
DEFAULT_IMGSZ = 320
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
FPS_SMOOTHING = 0.9


def find_weights(weights_path: Path) -> Path:
    """
    查找模型权重。

    除标准路径外，也兼容本次训练已经生成的嵌套路径：
    runs/detect/runs/detect/train/weights/best.pt
    """
    if weights_path.exists():
        return weights_path

    candidates = [
        ROOT / "runs" / "detect" / "runs" / "detect" / "train" / "weights" / "best.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    run_root = ROOT / "runs"
    if run_root.exists():
        matches = sorted(
            (p for p in run_root.rglob("best.pt")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if matches:
            return matches[0]

    raise FileNotFoundError(f"未找到权重文件：{weights_path}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="USB 摄像头实时目标检测")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help="模型权重路径，例如 runs/detect/train/weights/best.pt",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=DEFAULT_CAMERA,
        help="USB 摄像头编号，默认 0",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help="推理输入尺寸，默认 320",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
        help="置信度阈值，默认 0.25",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=DEFAULT_IOU,
        help="NMS 的 IOU 阈值，默认 0.45",
    )
    return parser.parse_args()


def open_camera(camera_index: int) -> cv2.VideoCapture:
    """
    打开摄像头。

    Windows 下优先尝试 CAP_DSHOW，通常更稳定一些。
    如果失败，再退回默认方式。
    """
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 {camera_index}")

    return cap


def main() -> None:
    """主函数：读取摄像头并实时推理。"""
    args = parse_args()
    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = ROOT / weights_path
    weights_path = find_weights(weights_path)

    model = YOLO(str(weights_path))
    use_half = torch.cuda.is_available()

    cap = open_camera(args.camera)

    # 可按需要设置分辨率，便于 USB 摄像头预览
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    window_name = "YOLO26 USB Camera View"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    smooth_fps = 0.0

    try:
        while True:
            start_time = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                print("摄像头读帧失败，正在重试...")
                continue

            # 直接对单帧做推理
            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=0 if use_half else "cpu",
                half=use_half,
                verbose=False,
            )

            # Ultralytics 会返回带框的可视化图像
            annotated = results[0].plot()

            elapsed = time.perf_counter() - start_time
            current_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            if smooth_fps == 0.0:
                smooth_fps = current_fps
            else:
                smooth_fps = FPS_SMOOTHING * smooth_fps + (1.0 - FPS_SMOOTHING) * current_fps

            fps_text = f"FPS: {smooth_fps:.1f}"
            cv2.putText(
                annotated,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
