#!/usr/bin/env bash

# 树莓派端启动脚本：通常只需要修改下面的参数。

# 电脑端运行 udp_viewer.py 的 IPv4 地址；填写电脑 ipconfig 显示的地址。
PC_IP="192.168.71.20"
UDP_PORT=8888
ENABLE_UDP=1             # 1=开启实时图传，0=关闭
UDP_FPS=30
UDP_QUALITY=60

# 模型默认放在仓库根目录：~/gedou/model.ncnn.param 和 model.ncnn.bin
MODEL_PARAM="model.ncnn.param"
MODEL_BIN="model.ncnn.bin"

SERIAL_PORT="/dev/ttyUSB0"
CAMERA_INDEX=0
INPUT_SIZE=320
YOLO_INTERVAL=3
TRACK_POINTS=6
HSV_RATIO=0.008
CONFIDENCE=0.25
NMS_THRESHOLD=0.45

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname -- "$SCRIPT_DIR")"
PROGRAM="$SCRIPT_DIR/build/robocup_infer"

if [[ ! -x "$PROGRAM" ]]; then
    echo "错误：找不到可执行程序：$PROGRAM" >&2
    echo "请先在树莓派上编译 C++ 程序。" >&2
    exit 1
fi

if [[ ! -f "$PROJECT_DIR/$MODEL_PARAM" ]]; then
    echo "错误：找不到模型结构文件：$PROJECT_DIR/$MODEL_PARAM" >&2
    exit 1
fi

if [[ ! -f "$PROJECT_DIR/$MODEL_BIN" ]]; then
    echo "错误：找不到模型权重文件：$PROJECT_DIR/$MODEL_BIN" >&2
    exit 1
fi

ARGS=(
    --model "$PROJECT_DIR/$MODEL_PARAM"
    --bin "$PROJECT_DIR/$MODEL_BIN"
    --serial "$SERIAL_PORT"
    --camera "$CAMERA_INDEX"
    --size "$INPUT_SIZE"
    --yolo-interval "$YOLO_INTERVAL"
    --track-points "$TRACK_POINTS"
    --hsv-ratio "$HSV_RATIO"
    --conf "$CONFIDENCE"
    --nms "$NMS_THRESHOLD"
)

if [[ "$ENABLE_UDP" == "1" ]]; then
    ARGS+=(
        --ip "$PC_IP"
        --port "$UDP_PORT"
        --udp-fps "$UDP_FPS"
        --udp-quality "$UDP_QUALITY"
    )
else
    ARGS+=(--no-udp)
fi

echo "启动程序：$PROGRAM"
echo "模型：$PROJECT_DIR/$MODEL_PARAM"
if [[ "$ENABLE_UDP" == "1" ]]; then
    echo "UDP 图传目标：$PC_IP:$UDP_PORT（${UDP_FPS} FPS）"
fi

exec "$PROGRAM" "${ARGS[@]}"
