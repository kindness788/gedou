# 轮式格斗视觉部署说明

本文说明如何把 YOLO 模型和 C++ 推理程序部署到树莓派，以及程序如何把目标结果发给下位机。shell 命令默认在项目仓库根目录执行；如果树莓派稀疏检出后当前目录就是 `C++`，请使用文中对应的目录写法。

## 程序做什么

摄像头提供画面，NCNN 在树莓派上运行 YOLO26n，OpenCV 负责增益块的 HSV 检查和光流跟踪。程序每个有效画面向下位机发送一个状态字节，分别表示两类块是否出现、各自的左右方向及距离阈值状态；也可以把标注后的画面通过 UDP 发到电脑查看。

类别编号固定为：

| 类别编号 | 模型名称 | 用途 |
| --- | --- | --- |
| `0` | `buff_block` | 增益块；没有减益块优先状态时作为跟踪目标 |
| `1` | `debuff_block` | 减益块；检测到后作为优先目标并停止增益块光流跟踪 |

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
  --distance-threshold 50 \
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
| `--distance-threshold` | `50` | 第 5 位的距离阈值；采用 `5000 / 目标框宽度` 的粗略距离量 |
| `--no-udp` | 默认不传此参数（UDP开启） | 使用该开关关闭图传 |
| `--show` | 关闭窗口 | 显示本机调试画面 |

当前摄像头采集请求为 `320×320`，但设备可能不接受该请求。输入模型前，程序会保持原始宽高比，把画面 letterbox 到 `320×320`，再送入 NCNN。摄像头采集尺寸与模型输入尺寸不是同一个参数。

如果串口设备不是 `/dev/ttyUSB0`，可先在树莓派查看设备：

```bash
ls /dev/ttyUSB* /dev/ttyACM* /dev/video*
```

当前程序串口打不开时会打印错误并继续运行，但串口数据不会送到下位机。必要时把当前用户加入系统的串口设备组，并重新登录后再运行。

## 识别、优先级和跟踪流程

```text
YOLO 检测到减益块
    ├─ 有增益块也优先选择减益块作为距离阈值判断目标
    ├─ 清空增益块光流状态
    ├─ 进入减益优先状态，逐帧运行 YOLO 更新减益块方向
    └─ 连续 3 帧没有检测到减益块后退出优先状态

没有减益块优先状态
    ├─ YOLO 每 3 帧检测一次增益块
    ├─ 相邻检测间隔用增益块 ROI 的 HSV 特征和 LK 光流更新位置
    └─ 光流失效时立即调用 YOLO；当前没有可跟踪目标时也逐帧搜索
```

多个同类别目标同时出现时，程序选检测框面积最大的一个。两类块同帧出现时，两种存在标志都会置位，并分别计算方向。光流只用于增益块；减益优先状态下不使用增益光流。退出减益优先状态时，若当前 YOLO 帧已经检测到增益块，会立即用它初始化增益跟踪。

调试画面中，红框是减益块，绿框是 YOLO 检测到的增益块，青框是光流更新的增益块。顶部状态会显示当前处于 YOLO 检测、增益跟踪或减益搜索状态。

## 串口数据格式

每个成功读取的非空摄像头画面发送一个字节，不再使用原先的 `AA ... BB` 帧头、帧尾、偏移量、距离数值与校验和。

```text
bit7 bit6 bit5 bit4 bit3 bit2 bit1 bit0
  0    0    0   距离  增益  减益  增益  减益
                达标  在左  在左  存在  存在
```

这里的“第 1 位”指最低位 `bit0`，下位机可直接按掩码解析：

| 位 | 掩码 | 含义 |
| --- | --- | --- |
| `bit0` | `0x01` | 有减益块为 `1`，无为 `0` |
| `bit1` | `0x02` | 有增益块为 `1`，无为 `0` |
| `bit2` | `0x04` | 减益块中心在画面左半边为 `1`，右半边或正中为 `0` |
| `bit3` | `0x08` | 增益块中心在画面左半边为 `1`，右半边或正中为 `0` |
| `bit4` | `0x10` | 优先目标的粗略距离达到阈值为 `1`，否则为 `0` |
| `bit5..7` | `0xE0` | 保留，发送时恒为 `0` |

某类不存在时，其方向位固定为 `0`。没有目标时整字节为 `0x00`。同帧检测到两类时，`bit0` 和 `bit1` 可同时为 `1`，方向位也分别有效。

距离阈值沿用原有估算量 `5000 / 目标框宽度（像素）`。当估算量小于或等于 `--distance-threshold` 时，`bit4` 置 `1`；没有目标时为 `0`。两类同时存在时以减益块为优先目标，仅有增益块时以增益块判断。默认阈值 `50` 相当于框宽至少约 `100` 像素；该量不是厘米，实机使用时需要标定阈值。下位机必须改为每次读取并解析单字节，旧版 8 字节协议解析器不能继续使用。

减益块由 YOLO 判断；增益块在两次 YOLO 之间可由光流更新。因此增益标志在光流帧仍可能有效，而减益标志只根据当前 YOLO 检测结果置位。发现减益块后程序逐帧运行 YOLO，直至连续 3 帧未发现减益块。

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
2. 确认减益块出现时画面显示红框，串口字节 `bit0` 为 `1`，`bit2` 与画面左右一致。
3. 确认仅有增益块时 `bit1` 为 `1`，`bit3` 与左右一致，画面能在 YOLO 框与光流框之间连续切换；两类同帧出现时 `bit0` 和 `bit1` 均为 `1`。
4. 用已知大小和距离的目标验证 `bit4` 阈值，再让下位机按单字节协议解析并进行低速避让/追踪测试。
5. 完成方向与串口联调后，再按需启用 UDP 图传和机器人控制。

调试时先验证类别和方向，再调 `--conf`、`--hsv-ratio` 等阈值。距离阈值需要实机标定后才能用于距离控制。
