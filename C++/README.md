# 轮式格斗视觉部署说明

本文说明如何把 YOLO 模型和 C++ 推理程序部署到树莓派，以及程序如何把目标结果发给下位机。shell 命令默认在项目仓库根目录执行；如果树莓派稀疏检出后当前目录就是 `C++`，请使用文中对应的目录写法。

## 程序做什么

摄像头提供画面，NCNN 在树莓派上运行 YOLO26n，OpenCV 负责增益块的 HSV 检查和光流跟踪。程序把当前目标类型、水平位置和粗略距离通过串口发给下位机，也可以把标注后的画面通过 UDP 发到电脑查看。

类别编号固定为：

| 类别编号 | 模型名称 | 用途 |
| --- | --- | --- |
| `0` | `buff_block` | 增益块；没有减益块优先状态时作为跟踪目标 |
| `1` | `debuff_block` | 减益块；检测到后优先上报并停止增益块光流跟踪 |

## 文件和模型

| 文件 | 用途 |
| --- | --- |
| `C++/main.cpp` | 摄像头采集、NCNN 推理、能量块优先级、HSV/光流跟踪、串口与 UDP |
| `C++/CMakeLists.txt` | C++ 程序的构建配置 |
| `runs/detect/train_clean/weights/best_ncnn_model/model.ncnn.param` | NCNN 网络结构 |
| `runs/detect/train_clean/weights/best_ncnn_model/model.ncnn.bin` | NCNN 模型权重 |
| `train_export.py` | 在训练电脑上训练 YOLO26n，并导出 NCNN 模型 |
| `robocup_data.yaml` | 数据集位置和类别配置 |

树莓派运行只需要 `main.cpp`、`CMakeLists.txt`、编译依赖以及一对匹配的 `.param` 和 `.bin` 模型文件。训练集、Python 训练脚本和 `.pt` 权重不需要复制到树莓派。

模型文件必须来自同一次导出，不能把不同目录或不同训练版本的 `.param` 与 `.bin` 混在一起。当前模型导出尺寸是 `320×320`，运行时 `--size` 必须保持 `320`。

## 从 GitHub 只检出 C++ 目录

第一次在树莓派克隆项目时，可以只把 `C++` 目录检出到工作目录：

```bash
git clone --filter=blob:none --no-checkout https://github.com/kindness788/gedou.git gedou
cd gedou
git sparse-checkout init --no-cone
git sparse-checkout set '/C++/'
git checkout main
```

以后更新已检出的代码：

```bash
cd ~/gedou
git pull origin main
```

稀疏检出只控制工作目录中显示哪些仓库文件，不会自动下载模型。因此还要把模型文件单独复制到树莓派。可以在 Windows PowerShell 中执行：

```powershell
scp D:\Projects\gedou_vision\runs\detect\train_clean\weights\best_ncnn_model\model.ncnn.param pi@树莓派IP:~/gedou/
scp D:\Projects\gedou_vision\runs\detect\train_clean\weights\best_ncnn_model\model.ncnn.bin pi@树莓派IP:~/gedou/
```

把 `pi` 和 `树莓派IP` 替换成实际 SSH 用户名和地址。复制后，树莓派上 `~/gedou/` 中应有 `C++/`、`model.ncnn.param` 和 `model.ncnn.bin`。

## 在树莓派编译

树莓派需要 C++14 编译器、CMake、OpenCV 开发库和 NCNN 开发库。先按当前 Raspberry Pi OS 版本安装这些依赖；如果 NCNN 安装在非系统默认位置，构建时通过 `ncnn_DIR`、`OpenCV_DIR` 或 `CMAKE_PREFIX_PATH` 告诉 CMake 库的位置。

如果当前目录是仓库根目录（例如 `~/gedou`）：

```bash
cmake -S C++ -B C++/build -DCMAKE_BUILD_TYPE=Release
cmake --build C++/build -j4
```

如果 VS Code 已打开 `C++` 子目录作为工作目录：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

编译成功后，程序位于 `C++/build/robocup_infer`（从仓库根目录构建）或 `build/robocup_infer`（从 `C++` 目录构建）。不要把 Windows 上编译出来的可执行文件复制到树莓派；必须在树莓派上为 Linux/ARM 架构编译。

## 运行推理

建议先关闭 UDP 图传，在摄像头、串口和目标策略确认后再接下位机控制：

```bash
./C++/build/robocup_infer \
  --model model.ncnn.param \
  --bin model.ncnn.bin \
  --serial /dev/ttyUSB0 \
  --size 320 \
  --conf 0.25 \
  --yolo-interval 3 \
  --track-points 6 \
  --hsv-ratio 0.008 \
  --no-udp
```

如果当前工作目录是 `C++`，把程序路径改为 `./build/robocup_infer`。有桌面环境并需要本地画面时可额外加 `--show`；没有桌面环境时不要加这个参数。

常用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | `model.ncnn.param` | NCNN 网络结构文件路径 |
| `--bin` | `model.ncnn.bin` | 对应的 NCNN 权重文件路径 |
| `--serial` | `/dev/ttyUSB0` | 下位机串口设备；按实际设备名修改 |
| `--baud` | `115200` | 串口波特率 |
| `--camera` | `0` | 摄像头编号，默认 `/dev/video0` 对应的索引 |
| `--size` | `320` | YOLO 输入边长；必须与当前导出的模型匹配 |
| `--conf` | `0.25` | 检测置信度阈值 |
| `--nms` | `0.45` | NMS 重叠阈值 |
| `--yolo-interval` | `3` | 增益跟踪时 YOLO 校正间隔，单位为摄像头帧 |
| `--track-points` | `6` | 光流跟踪所需的最少有效特征点数 |
| `--hsv-ratio` | `0.008` | 增益块 ROI 的最低颜色占比阈值 |
| `--no-udp` | 默认不传此参数（UDP开启） | 使用该开关关闭图传 |
| `--show` | 关闭窗口 | 显示本机调试画面 |

当前摄像头采集请求为 `320×240`，但设备可能不接受该请求。输入模型前，程序会保持原始宽高比，把画面 letterbox 到 `320×320`，再送入 NCNN。摄像头采集尺寸与模型输入尺寸不是同一个参数。

如果串口设备不是 `/dev/ttyUSB0`，可先在树莓派查看设备：

```bash
ls /dev/ttyUSB* /dev/ttyACM* /dev/video*
```

当前程序串口打不开时会打印错误并继续运行，但串口数据不会送到下位机。必要时把当前用户加入系统的串口设备组，并重新登录后再运行。

## 识别、优先级和跟踪流程

```text
YOLO 检测到减益块
    ├─ 有增益块也优先选择减益块
    ├─ 清空增益块光流状态
    ├─ 进入减益优先状态，逐帧运行 YOLO 更新减益块方向
    └─ 连续 3 帧没有检测到减益块后退出优先状态

没有减益块优先状态
    ├─ YOLO 每 3 帧检测一次增益块
    ├─ 相邻检测间隔用增益块 ROI 的 HSV 特征和 LK 光流更新位置
    └─ 光流失效时立即调用 YOLO；当前没有可跟踪目标时也逐帧搜索
```

多个同类别目标同时出现时，程序选检测框面积最大的一个。光流只用于增益块；减益优先状态下不使用增益光流。退出减益优先状态时，若当前 YOLO 帧已经检测到增益块，会立即用它初始化增益跟踪。

调试画面中，红框是减益块，绿框是 YOLO 检测到的增益块，青框是光流更新的增益块。顶部状态会显示当前处于 YOLO 检测、增益跟踪或减益搜索状态。

## 串口数据格式

每帧发送一个 8 字节包：

```text
AA type dx_hi dx_lo distance_hi distance_lo checksum BB
```

| 字节 | 含义 |
| --- | --- |
| `0` | 帧头 `0xAA` |
| `1` | 目标类型：`0x00` 无目标，`0x01` 增益块，`0x02` 减益块 |
| `2..3` | `dx` 有符号 16 位整数，大端补码 |
| `4..5` | `distance` 无符号 16 位整数，大端 |
| `6` | 校验和：字节 `1` 到 `5` 相加后取低 8 位 |
| `7` | 帧尾 `0xBB` |

`dx` 是目标框中心相对画面中心的水平像素偏移：负数表示目标在画面左侧，正数表示在画面右侧，`0` 表示中心附近。下位机应先按 `type` 区分增益和减益目标，再读取 `dx`；减益块需要根据方向避让。

`distance` 当前按 `round(5000 / 目标框宽度)` 计算，只是与框宽相关的粗略量，不是厘米。没有目标时，类型、`dx` 和 `distance` 都发为 `0`。修改本协议后，下位机的接收与校验代码也必须同步更新；旧程序如果把 `type` 当作简单的 `found` 布尔值，就无法区分增益和减益。

## 可选：UDP 图传调试

默认 UDP 目标为 `192.168.139.200:8888`。启用图传时，需要把 IP 改成运行查看器的电脑地址，并在完整项目仓库的电脑端运行：

```powershell
python udp_viewer.py --bind-ip 0.0.0.0 --bind-port 8888
```

树莓派运行时去掉 `--no-udp`，并设置目标地址：

```bash
./C++/build/robocup_infer \
  --model model.ncnn.param --bin model.ncnn.bin \
  --serial /dev/ttyUSB0 --size 320 \
  --ip 运行查看器的电脑IP --port 8888
```

图传使用 UDP 分片 JPEG，网络不稳定时可能丢帧；串口目标数据与图传相互独立。

## 训练电脑上的模型更新

训练和评估脚本只在包含数据集与 Python 文件的完整项目仓库中使用，不需要放到树莓派：

```powershell
python dataset_audit.py --write-splits --force
python train_export.py
python evaluate_model.py --weights runs/detect/train_clean/weights/best.pt
python compare_backends.py
```

`train_export.py` 使用 `320` 输入尺寸，训练完成后导出 NCNN。`compare_backends.py` 用验证图像对照 Ultralytics NCNN 后端，检查 C++ 采用的预处理和解码。导出新模型后，把同一导出目录内的 `model.ncnn.param` 和 `model.ncnn.bin` 一起复制到树莓派。

## 初次调试顺序

1. 先用 `--show --no-udp` 确认摄像头图像正常、类别和框的位置正确。
2. 确认减益块出现时画面显示红框，串口目标类型为 `0x02`，`dx` 的正负方向与画面左右一致。
3. 确认仅有增益块时类型为 `0x01`，画面能在 YOLO 框与光流框之间连续切换；移动或遮挡后能重新检测。
4. 确认下位机按新串口协议解析类型、偏移和校验和，再进行低速避让/追踪测试。
5. 完成方向与串口联调后，再按需启用 UDP 图传和机器人控制。

调试时先验证类别和方向，再调 `--conf`、`--hsv-ratio` 等阈值。距离值需要实机标定后才能用于距离控制。
