"""Verify custom NCNN preprocessing/decoding against Ultralytics NCNN output."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import ncnn
import numpy as np


ROOT = Path(__file__).resolve().parent

from ultralytics import YOLO  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查自定义 NCNN 解码与官方后端是否一致")
    parser.add_argument("--ncnn", type=Path, default=ROOT / "runs/detect/train_clean/weights/best_ncnn_model")
    parser.add_argument("--manifest", type=Path, default=ROOT / "dataset/robocup/splits/val.txt")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--min-match-iou", type=float, default=0.95)
    return parser.parse_args()


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    pad_x = (size - resized_width) // 2
    pad_y = (size - resized_height) // 2
    resized = cv2.resize(image, (resized_width, resized_height))
    output = np.full((size, size, 3), 114, dtype=np.uint8)
    output[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    return output, scale, pad_x, pad_y


def nms_per_class(boxes, scores, classes, conf, iou) -> np.ndarray:
    keep: list[int] = []
    for class_id in np.unique(classes):
        source = np.flatnonzero(classes == class_id)
        xywh = boxes[source].copy()
        xywh[:, 2:] -= xywh[:, :2]
        indices = cv2.dnn.NMSBoxes(xywh.tolist(), scores[source].tolist(), conf, iou)
        keep.extend(int(source[int(index)]) for index in np.asarray(indices).reshape(-1))
    return np.asarray(keep, dtype=np.int64)


def decode_raw(raw, image_shape, scale, pad_x, pad_y, conf, iou):
    if raw.ndim != 2:
        raise RuntimeError(f"NCNN 输出应为二维，实际为 {raw.shape}")
    if raw.shape[0] > raw.shape[1]:
        raw = raw.T
    if raw.shape[0] < 6:
        raise RuntimeError(f"NCNN 输出属性数不足：{raw.shape}")

    class_scores = raw[4:]
    classes = np.argmax(class_scores, axis=0).astype(np.int64)
    scores = class_scores[classes, np.arange(raw.shape[1])]
    selected = scores >= conf
    xywh = raw[:4, selected].T
    scores = scores[selected]
    classes = classes[selected]
    if not len(xywh):
        return np.empty((0, 4)), classes, scores

    boxes = np.empty_like(xywh)
    boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    height, width = image_shape
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height - 1)
    valid = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
    boxes, scores, classes = boxes[valid], scores[valid], classes[valid]
    keep = nms_per_class(boxes, scores, classes, conf, iou)
    return boxes[keep], classes[keep], scores[keep]


def box_iou(one: np.ndarray, many: np.ndarray) -> np.ndarray:
    left_top = np.maximum(one[:2], many[:, :2])
    right_bottom = np.minimum(one[2:], many[:, 2:])
    intersection = np.prod(np.clip(right_bottom - left_top, 0.0, None), axis=1)
    one_area = np.prod(np.clip(one[2:] - one[:2], 0.0, None))
    many_area = np.prod(np.clip(many[:, 2:] - many[:, :2], 0.0, None), axis=1)
    return intersection / np.clip(one_area + many_area - intersection, 1e-9, None)


def main() -> int:
    args = parse_args()
    param = args.ncnn / "model.ncnn.param"
    weights = args.ncnn / "model.ncnn.bin"
    for path in (param, weights, args.manifest):
        if not path.exists():
            raise FileNotFoundError(path)

    images = [Path(line.strip()) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()][: args.limit]
    reference_model = YOLO(str(args.ncnn), task="detect")
    net = ncnn.Net()
    if net.load_param(str(param)) != 0 or net.load_model(str(weights)) != 0:
        raise RuntimeError("无法加载 NCNN 模型")

    matched = unmatched = 0
    match_ious: list[float] = []
    score_deltas: list[float] = []
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"无法读取图片：{image_path}")
        prepared, scale, pad_x, pad_y = letterbox(image, args.imgsz)
        rgb = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB)
        chw = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
        with net.create_extractor() as extractor:
            extractor.input("in0", ncnn.Mat(chw).clone())
            return_code, output = extractor.extract("out0")
        if return_code != 0:
            raise RuntimeError(f"NCNN 推理失败：{return_code}")
        boxes, classes, scores = decode_raw(np.array(output), image.shape[:2], scale, pad_x, pad_y, args.conf, args.iou)

        reference = reference_model.predict(str(image_path), imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)[0]
        ref_boxes = reference.boxes.xyxy.cpu().numpy()
        ref_classes = reference.boxes.cls.cpu().numpy().astype(np.int64)
        ref_scores = reference.boxes.conf.cpu().numpy()
        used: set[int] = set()
        for box, class_id, score in zip(boxes, classes, scores):
            candidates = [idx for idx in np.flatnonzero(ref_classes == class_id) if int(idx) not in used]
            if not candidates:
                unmatched += 1
                continue
            candidate_array = np.asarray(candidates, dtype=np.int64)
            candidate_ious = box_iou(box, ref_boxes[candidate_array])
            best_position = int(np.argmax(candidate_ious))
            best_index = int(candidate_array[best_position])
            best_iou = float(candidate_ious[best_position])
            if best_iou < args.min_match_iou:
                unmatched += 1
                continue
            used.add(best_index)
            matched += 1
            match_ious.append(best_iou)
            score_deltas.append(abs(float(score) - float(ref_scores[best_index])))
        unmatched += len(ref_boxes) - len(used)

    print(f"images={len(images)}, matched={matched}, unmatched={unmatched}")
    if match_ious:
        print(f"IoU min/mean={min(match_ious):.6f}/{np.mean(match_ious):.6f}, confidence delta max/mean={max(score_deltas):.6f}/{np.mean(score_deltas):.6f}")
    if unmatched:
        print("自定义解码与 Ultralytics NCNN 后端不一致。")
        return 2
    print("自定义 NCNN 解码与参考后端一致。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
