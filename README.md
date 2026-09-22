# 轮式格斗视觉

当前视觉链路使用 YOLO26n 识别增益/减益能量块，训练后导出 NCNN，
在树莓派端通过 OpenCV + NCNN 推理，并通过串口给下位机发送目标数据。

训练与评估固定使用 Ultralytics `8.4.21`。YOLO26 的端到端分支和导出后处理
在不同 Ultralytics 版本间存在差异，请勿混用版本比较指标。安装前先根据显卡
安装合适的 PyTorch，再执行：

```powershell
pip install -r requirements-train.txt
```

## 数据准备

数据划分按拍摄日期隔离，避免同一段视频的相邻帧同时进入训练和验证：

```powershell
python dataset_audit.py --write-splits --force
```

该命令只生成 `dataset/robocup/splits/train.txt` 和 `val.txt`，不会移动或
删除原始图片。每次新增数据后都应重新生成并检查输出。真正的测试集应
来自未来独立采集场次，不参与模型和阈值选择。

当前类别只有：

- `0 buff_block`
- `1 debuff_block`

允许用空标签文件表示纯背景负样本。建议补充赛台、对手机器人、出发区、
围栏和人员等无能量块画面。

详细场景数量和标注约定见 `DATA_COLLECTION.md`。

## 训练与评估

```powershell
python train_export.py
python evaluate_model.py
python compare_backends.py
```

训练脚本会先运行数据审计；审计不通过时不会启动训练。评估结果写入
`runs/detect/clean_val`，并分别输出两个类别的 mAP。
`compare_backends.py` 会用同一批图片检查自定义 NCNN 预处理/解码与
Ultralytics NCNN 参考后端的框、类别和置信度。

## 树莓派构建

在树莓派上安装 OpenCV、NCNN 和 CMake 后：

```bash
cmake -S C++ -B C++/build -DCMAKE_BUILD_TYPE=Release
cmake --build C++/build -j4
```

若库不在系统默认路径，通过 `OpenCV_DIR`、`ncnn_DIR` 或
`CMAKE_PREFIX_PATH` 传入，不要修改 `CMakeLists.txt` 写死开发机路径。

运行示例：

```bash
./C++/build/robocup_infer \
  --model model.ncnn.param --bin model.ncnn.bin \
  --serial /dev/ttyUSB0 --size 320 --conf 0.25 \
  --ip 192.168.139.200 --port 8888
```

比赛时不需要图传可加 `--no-udp`，本机调试窗口使用 `--show`。

## 图传查看

```powershell
python udp_viewer.py --bind-ip 0.0.0.0 --bind-port 8888
```

查看器支持 C++ 端使用的 `GDU1` 分片协议，也兼容单个 UDP 包内的完整 JPEG。

## 串口协议

当前保持原有 8 字节协议，避免破坏下位机兼容性：

```text
AA found dx_hi dx_lo distance_hi distance_lo checksum BB
```

其中 `distance` 仍是基于框宽的粗略量，不能视为厘米。正式控制前应使用
实机数据完成距离标定，或改用相机地面单应性。
