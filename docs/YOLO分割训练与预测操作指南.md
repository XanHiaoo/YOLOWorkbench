# YOLO 分割训练与预测操作指南

> 数据集：`your-segmentation-dataset`（裂缝分割）
> 模型：YOLO11n-seg
> 环境：Windows + WSL2 + Docker Desktop（CUDA 12.8 + RTX 5060 Ti）

---

## 目录

1. [项目结构](#一项目结构)
2. [数据集准备与整理](#二数据集准备与整理)
3. [Docker 环境配置](#三docker-环境配置)
4. [训练](#四训练)
5. [查看训练情况](#五查看训练情况)
6. [预测（picture2 推理）](#六预测picture2-推理)
7. [查看预测情况](#七查看预测情况)
8. [常见问题 FAQ](#八常见问题-faq)
9. [本次训练结果汇总](#九本次训练结果汇总)

---

## 一、项目结构

```
YOLO/
├── datasets/
│   ├── your-segmentation-dataset/
│   │   ├── data.yaml          # 数据集配置（已修复）
│   │   ├── Train.txt          # 原始训练清单（路径有误，已弃用）
│   │   ├── images/Train/      # 训练图片（100 张 .jpg）
│   │   └── labels/Train/      # 分割标注（52 个 .txt）
│   └── picture2/              # 验证集（仅图片，约 2000 张，无标签）
├── docker/
│   ├── Dockerfile             # 开发环境镜像定义
│   ├── run.ps1                # Windows PowerShell 启动脚本
│   └── run_wsl.sh             # WSL/Linux 启动脚本（新增）
├── models/                    # 预训练权重（yolo11n.pt 等）
├── runs/
│   ├── segment/segmentation/     # 训练输出（权重、曲线、CSV）
│   └── predict/inference-results/ # 推理输出（标注图）
└── python/                    # Python 脚本
```

---

## 二、数据集准备与整理

### 2.1 数据集原始状态

把数据集文件夹（含 `data.yaml`、`images/`、`labels/`）和验证集文件夹 `picture2` 放到项目根目录的 `datasets/` 下，最终路径为：

```
datasets/your-segmentation-dataset/
datasets/your-inference-images/
```

### 2.2 发现并修复的 3 个问题

原始 `data.yaml` 内容如下（**存在 2 个问题**）：

```yaml
Train: Train.txt      # 问题1：键名应是小写 train/val
names:
  0: crack
path: .               # 问题2：相对路径 "." 依赖运行目录
```

| # | 问题 | 说明 | 修复 |
|---|------|------|------|
| 1 | `Train:` 大写键名 | Ultralytics 标准用 `train:` / `val:` | 改为小写 |
| 2 | `Train.txt` 路径错误 | 文件内容写的是 `data/images/Train/1.jpg`，但真实目录是 `images/Train/`（多了一个 `data/` 前缀），按此清单训练会找不到图片 | 弃用 Train.txt，`train` 直接指向目录 `images/Train` |
| 3 | 没有 val 验证集 | Ultralytics 训练必须配置 `val`，否则报错 | 先用 `images/Train` 占位（注意：这样评估的 mAP 会偏乐观） |

### 2.3 修复后的 `data.yaml`

```yaml
path: datasets/your-segmentation-dataset
train: images/Train
val: images/Train
names:
  0: crack
```

> `path` 使用相对路径，在项目根目录（宿主机 `/mnt/c/path/to/YOLOWorkbench` 或容器 `/workspace`）执行命令即可正确解析。

### 2.4 验证集 `picture2` 的定位

- 目录：`datasets/your-inference-images/`（约 2000 张 jpg，编号 2001~4000）
- **只有图片，没有标签文件** → 无法作为训练验证集（val 需要标签计算 mAP）
- 定位：**训练完成后的推理测试集**（predict 阶段使用，无需标签）

### 2.5 数据量核实

| 目录 | 数量 | 说明 |
|------|------|------|
| `images/Train/` | 100 张 | 编号 1~1088 不连续抽样 |
| `labels/Train/` | 52 个 | 仅 52 张图有标注 |
| `picture2/` | ~2000 张 | 无标签 |

> 提示：100 张训练图里仅 52 张有标注，数据量偏小，是最终指标的上限瓶颈之一。

---

## 三、Docker 环境配置

### 3.1 为什么必须用 Docker

宿主机 WSL 上没有安装 Ultralytics（`yolo` 命令不存在），开发环境统一在 Docker 镜像 `yoloworkbench:cuda12.8` 中（含 Python 3.12 + PyTorch CUDA 12.8 + Ultralytics）。

### 3.2 检查镜像

```bash
docker images | grep yolo
```

如果不存在，先构建：

```bash
cd /mnt/c/path/to/YOLOWorkbench
docker build -t yoloworkbench:cuda12.8 -f docker/Dockerfile .
```

### 3.3 启动脚本 `docker/run_wsl.sh`（新增）

原始 `run.ps1` 是 PowerShell 脚本，**无法在 WSL bash 中直接运行**（会报 `syntax error`）。因此新增了 bash 版启动脚本 `docker/run_wsl.sh`：

```bash
#!/bin/bash
set -e
NAME="${CONTAINER_NAME:-yolo-dev}"
IMAGE="${IMAGE:-yoloworkbench:cuda12.8}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# 检查 docker / 镜像
if ! command -v docker >/dev/null 2>&1; then echo "[ERROR] docker not found"; exit 1; fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then echo "[ERROR] image not found"; exit 1; fi

# --recreate：强制重建
if [ "$1" = "--recreate" ]; then docker rm -f "$NAME" 2>/dev/null || true; fi

# 容器不存在则创建（后台常驻），已存在则复用
if ! docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    docker run -d --name "$NAME" --gpus all --shm-size=8g \
        -v "$PROJECT_ROOT:/workspace" -w /workspace "$IMAGE" sleep infinity
elif [ "$(docker inspect -f '{{.State.Running}}' "$NAME")" != "true" ]; then
    docker start "$NAME"
fi

# 进入容器
docker exec -it "$NAME" bash
```

使用方式：

```bash
chmod +x /mnt/c/path/to/YOLOWorkbench/docker/run_wsl.sh   # 首次需赋予执行权限
cd /mnt/c/path/to/YOLOWorkbench
./docker/run_wsl.sh                # 启动/进入容器
./docker/run_wsl.sh --recreate     # 强制删除并重建容器
```

### 3.4 容器启动关键参数（重要）

```bash
docker run -d --name yolo-dev \
  --gpus all \                      # GPU 透传（RTX 5060 Ti）
  --shm-size=8g \                   # 共享内存 8G（默认 64MB 会导致数据加载失败！）
  -v "/mnt/c/path/to/YOLOWorkbench:/workspace" \
  -w /workspace \
  yoloworkbench:cuda12.8 sleep infinity
```

| 参数 | 作用 | 缺失后果 |
|------|------|---------|
| `--gpus all` | 使用 GPU | 训练极慢或报 CUDA 不可用 |
| `--shm-size=8g` | 加大共享内存 | PyTorch 多进程加载数据报 `unable to allocate shared memory` |
| `-d` + `sleep infinity` | 后台常驻 | 无 |
| `-v ...:/workspace` | 目录挂载 | 容器内看不到项目文件 |

> **坑点**：不要用 `docker run -it` 直接进入（容器主进程是 `sleep infinity`，不读终端输入，看起来"卡住无反应"）。正确姿势是**先 `-d` 后台启动，再用 `docker exec -it` 进入**。

### 3.5 验证环境

```bash
docker exec yolo-dev bash -c "python3 --version && python3 -c 'import ultralytics; print(ultralytics.__version__)'"
# 预期输出：Python 3.12.3 / 8.4.120

docker exec yolo-dev python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
# 预期输出：CUDA: True NVIDIA GeForce RTX 5060 Ti
```

---

## 四、训练

### 4.1 训练命令

在项目根目录执行（容器内 `/workspace` 与宿主机 `/mnt/c/path/to/YOLOWorkbench` 是同一目录）：

```bash
docker exec yolo-dev yolo segment train \
    data="datasets/your-segmentation-dataset/data.yaml" \
    model=yolo11n-seg.pt \
    epochs=200 \
    imgsz=640 \
    batch=16 \
    patience=50 \
    device=0 \
    name=segmentation
```

### 4.2 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `data` | data.yaml 路径 | 数据集配置（含 train/val/names） |
| `model` | `yolo11n-seg.pt` | 分割预训练权重，首次运行自动下载；也可用 `yolo11s-seg.pt` 等更大模型 |
| `epochs` | 200 | 最大训练轮数 |
| `imgsz` | 640 | 输入图片尺寸 |
| `batch` | 16 | 每批数量，按显存调整（16G 显存可用 16） |
| `patience` | 50 | 验证指标 50 轮无提升即早停 |
| `device` | 0 | GPU 编号；无 GPU 改 `cpu` |
| `name` | segmentation | 输出目录名 |

> 训练输出默认保存到 `runs/segment/segmentation/`。注意 segment 任务默认 project 就是 `runs/segment`，**不要再重复指定 `project=runs/segment`**（否则路径会嵌套成 `runs/segment/runs/segment/...`）。

### 4.3 训练启动成功的标志

日志中出现：

```
Ultralytics 8.4.120 🚀 Python-3.12.3 torch-2.11.0+cu128 CUDA:0 (NVIDIA GeForce RTX 5060 Ti, 16311MiB)
...
      Epoch    GPU_mem   box_loss   seg_loss   ...   1/200 ...
```

### 4.4 训练输出目录

```
runs/segment/segmentation/
├── args.yaml                    # 训练参数存档
├── results.csv                  # 每轮指标（机器可读）
├── results.png                  # 训练曲线总图
├── weights/
│   ├── best.pt                  # 验证集最优权重（部署用它）
│   └── last.pt                  # 最后一轮权重
├── train_batch*.jpg             # 训练批次可视化
├── val_batch*_pred.jpg          # 验证集预测可视化
├── val_batch*_labels.jpg        # 验证集标注可视化
├── confusion_matrix*.png        # 混淆矩阵
├── Box*.png / Mask*.png         # P/R/PR/F1 曲线
└── labels.jpg                   # 数据集标注分布图
```

### 4.5 `segmentation/` 目录逐文件详解

训练完成后，`runs/segment/segmentation/` 目录下会生成以下文件，逐个说明：

| 文件 | 类型 | 内容说明 | 用途 |
|------|------|---------|------|
| `args.yaml` | 配置 | 本次训练的全部参数快照 | 复现训练、排查参数 |
| `results.csv` | 数据 | 每个 epoch 的完整指标（loss / mAP / lr） | 精确数值分析、画自定义曲线 |
| `results.png` | 图表 | 训练全程的总曲线（多子图） | **首选看这张**，判断收敛情况 |
| `weights/best.pt` | 权重 | 验证集指标最好的模型 | **部署/推理用它** |
| `weights/last.pt` | 权重 | 最后一个 epoch 的模型 | 继续训练/对比用 |
| `train_batch*.jpg` | 图片 | 训练批次（含数据增强后的图） | 检查输入是否正常 |
| `val_batch*_labels.jpg` | 图片 | 验证集标注（ground truth） | 对照预测看效果 |
| `val_batch*_pred.jpg` | 图片 | 验证集模型预测结果 | **看模型实际预测效果** |
| `confusion_matrix.png` | 图表 | 混淆矩阵（计数版） | 看分类混淆情况 |
| `confusion_matrix_normalized.png` | 图表 | 混淆矩阵（归一化版） | 百分比形式更直观 |
| `BoxP/R/PR/F1_curve.png` | 图表 | 检测框（Box）的 P / R / PR / F1 曲线 | 分析检测框阈值 |
| `MaskP/R/PR/F1_curve.png` | 图表 | 分割掩码（Mask）的 P / R / PR / F1 曲线 | 分析分割阈值 |
| `labels.jpg` | 图片 | 数据集标注分布总览（所有标注画在一张图上） | 看标注质量与分布 |

### 4.6 `results.png` —— 训练总曲线（图文解读）

这是最重要的图，位于 `runs/segment/segmentation/results.png`：

（本机训练/验证结果示意图未随仓库提交。）

该图由多行子图组成，Ultralytics 8.x 的布局为：

| 行 | 子图内容 | 解读要点 |
|----|---------|---------|
| 第 1 行 | train/box_loss、train/seg_loss、train/cls_loss、train/dfl_loss | 训练损失，**应持续下降并趋于平稳** |
| 第 2 行 | val/box_loss、val/seg_loss、val/cls_loss、val/dfl_loss | 验证损失，**不应明显反弹**（反弹=过拟合） |
| 第 3 行 | metrics/precision(B)、metrics/recall(B)、mAP50(B)、mAP50-95(B) | 检测框指标，**应上升并变平** |
| 第 4 行 | metrics/precision(M)、metrics/recall(M)、mAP50(M)、mAP50-95(M) | 分割掩码指标（**裂缝分割重点看这里**） |
| 第 5 行 | lr/pg0、lr/pg1、lr/pg2 | 学习率变化（cos_lr 时为余弦衰减曲线） |

**本次训练实测趋势**（来自 `results.csv`）：

| epoch | train/box_loss | train/seg_loss | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
|------:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 1.272 | 5.226 | 0.075 | 0.031 | 0.008 | 0.001 |
| 50 | 1.939 | 1.211 | 0.065 | 0.023 | 0.082 | 0.025 |
| 100 | 1.518 | 1.067 | 0.286 | 0.127 | 0.333 | 0.113 |
| 125 | 1.373 | 0.941 | 0.577 | 0.265 | 0.582 | 0.238 |
| 150 | 1.191 | 1.017 | 0.647 | 0.310 | 0.655 | 0.263 |
| 175 | 1.039 | 0.787 | 0.705 | 0.367 | 0.711 | 0.325 |
| 200 | 1.095 | 0.583 | 0.754 | 0.449 | 0.756 | 0.396 |

**解读**：
- `seg_loss` 从 5.23 一路降到 0.58，收敛明显
- Mask mAP50 从 0.008 涨到 0.756，训练**始终在提升、未过拟合**（200 轮跑满且无早期停顿）
- 各指标曲线在 100~200 轮段仍缓慢上升，说明若继续训练还有提升空间，也可加大 epoch 数

### 4.7 置信度曲线 `Box/Mask P/R/PR/F1`（图文解读）

这 8 张图（`runs/segment/segmentation/Box*.png` / `Mask*.png`）结构相同，以 `BoxPR_curve.png` 为例：

（本机训练/验证结果示意图未随仓库提交。）

| 文件 | 横轴 | 纵轴 | 含义 |
|------|------|------|------|
| `*P_curve.png` | Confidence（置信度阈值） | Precision（精确率） | 阈值越高，Precision 越高 |
| `*R_curve.png` | Confidence | Recall（召回率） | 阈值越高，Recall 越低 |
| `*PR_curve.png` | Recall | Precision | **PR 曲线下的面积 ≈ AP**，越接近 1 越好 |
| `*F1_curve.png` | Confidence | F1 分数 | **F1 最高点对应最佳置信度阈值**，用于推理时选 conf |

**用法**：部署推理时，可以看 `MaskF1_curve.png` 中 F1 峰值对应的置信度，把它设为 `conf` 参数，能取得召回/精确的最佳平衡。

### 4.8 混淆矩阵（图文解读）

`confusion_matrix.png`（计数）与 `confusion_matrix_normalized.png`（百分比）：

（本机训练/验证结果示意图未随仓库提交。）

- 横轴 = 预测类别（Predicted），纵轴 = 真实类别（True）
- 本数据集只有 1 类 `crack`，矩阵是 2×2（crack + background）
- 重点看：**主对角线值**（预测正确占比）；右上角 background 被误判为 crack 的占比 = 误检率

> 提示：因 val 用的是训练集本身，这里的混淆矩阵偏乐观，真实部署场景参考 picture2 推理结果。

### 4.9 `labels.jpg` —— 标注分布总览

`runs/segment/segmentation/labels.jpg` 把数据集里**所有标注**叠加绘制：

- 左上：所有 mask 归一化到同一画布的**分布热区**（可看出裂缝在图片中的常见位置）
- 其余：实例数量、类别数量、mask 面积分布直方图等

**解读重点**：如果 mask 分布集中在一小片区域，说明数据场景单一，泛化会受限；本数据集分布越分散，泛化潜力越好。

### 4.10 批次可视化图（图文解读）

`train_batch0.jpg` 展示增强后的训练输入；`val_batch0_labels.jpg` 是验证集标注、`val_batch0_pred.jpg` 是模型预测：

（本机训练/验证结果示意图未随仓库提交。）

（本机训练/验证结果示意图未随仓库提交。）

**解读方法**：左右对比 `val_batch*_labels` 与 `val_batch*_pred`：
- 绿色掩码是否贴合裂缝走向（形态相似性）
- 有无漏检（label 有但 pred 没有）与误检（pred 有但 label 没有）

### 4.11 关键结论

1. **训练健康**：loss 下降、mAP 上升、无过拟合迹象，200 轮合理
2. **最佳权重**：`weights/best.pt`（Mask mAP50 = 0.756）
3. **注意局限**：因 val 与 train 同源，指标偏乐观；真实效果以 picture2 推理为准
4. **若需提升**：补足 48 张无标注图的标签、增大训练集、换 `yolo11s-seg.pt` 更大模型、按 F1 曲线选取更优 conf

---

## 五、查看训练情况

### 5.1 实时滚动查看指标（推荐）

```bash
docker exec yolo-dev tail -f /workspace/runs/segment/segmentation/results.csv
```

按 `Ctrl+C` 退出。CSV 每行一个 epoch，列含义：

```
epoch, time, train/box_loss, train/seg_loss, train/cls_loss, train/dfl_loss,
metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B),
metrics/precision(M), metrics/recall(M), metrics/mAP50(M), metrics/mAP50-95(M), ...
```

- `(B)` = Box（检测框指标），`(M)` = Mask（分割掩码指标，**裂缝分割主要看这个**）
- mAP50 越高越好，loss 越低越好

### 5.2 查看训练曲线图

宿主机路径直接打开（IDE/图片查看器）：

```
/mnt/c/path/to/YOLOWorkbench/runs/segment/segmentation/results.png
```

### 5.3 查看权重文件

```bash
ls -la /mnt/c/path/to/YOLOWorkbench/runs/segment/segmentation/weights/
# best.pt 为最优权重，部署/推理都用它
```

### 5.4 检查 GPU 占用确认在训练

```bash
docker exec yolo-dev nvidia-smi
```

### 5.5 查看当前轮次

```bash
docker exec yolo-dev tail -1 /workspace/runs/segment/segmentation/results.csv
```

---

## 六、预测（picture2 推理）

### 6.1 预测命令

```bash
docker exec yolo-dev yolo segment predict \
    model=/workspace/runs/segment/segmentation/weights/best.pt \
    source="/workspace/datasets/your-inference-images" \
    imgsz=640 \
    conf=0.25 \
    save=True \
    project=/workspace/runs/predict \
    name=inference-results
```

### 6.2 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `model` | `best.pt` 绝对路径 | 用训练好的最优权重 |
| `source` | `picture2` 目录 | 推理整个目录（也可指定单张图/视频） |
| `conf` | 0.25 | 置信度阈值，越低检出越多（弱裂缝） |
| `save` | True | 保存标注后的图片 |
| `project/name` | runs/predict / inference-results | 输出目录 |

### 6.3 预测输出

```
runs/predict/inference-results/
├── 2001.jpg    # 检测到裂缝 → 带绿色 mask 标注
├── 2002.jpg
└── ...
```

---

## 七、查看预测情况

### 7.1 查看标注结果图

宿主机直接打开输出目录：

```
/mnt/c/path/to/YOLOWorkbench/runs/predict/inference-results/
```

有检测的图片带绿色分割掩码；无检测的图片仅保存原图。

### 7.2 从推理日志解读

推理时控制台会逐张输出：

```
image 1/2000 /workspace/datasets/your-inference-images/2001.jpg: 640x640 2 cracks, 14.4ms
image 32/2000 /workspace/datasets/your-inference-images/2032.jpg: 640x640 (no detections), 14.3ms
```

- `2 cracks` → 检到 2 条裂缝
- `(no detections)` → 未检出（可能是图中无裂缝，也可能是阈值过高漏检）
- 结尾的 `Speed: ... 17.3ms inference ...` 是平均单张耗时

### 7.3 统计检测情况（示例）

```bash
# 统计输出目录中哪些图有标注、哪些只有原图，可用文件大小区分
ls -la /mnt/c/path/to/YOLOWorkbench/runs/predict/inference-results/ | awk '{print $5, $9}' | sort -n | head
```

### 7.4 如何判断模型好坏

- 有裂缝的图：mask 是否完整贴合裂缝轮廓
- 无裂缝的图：是否出现误检（假阳性）
- 若某批次图片大量 `no detections`：可尝试 `conf=0.1` 重跑，判断是漏检还是图中确实无裂缝

---

## 八、常见问题 FAQ

### Q1：宿主机执行 `yolo` 报 `Command 'yolo' not found`？
**A**：`yolo` 只在 Docker 容器内可用。先 `./docker/run_wsl.sh` 进入容器再执行，或在宿主机上用 `docker exec yolo-dev yolo ...`。

### Q2：`docker run -it` 启动后"没反应"？
**A**：容器主进程是 `sleep infinity`，不读终端输入，`-it` 附加看起来像卡死。**容器实际已启动**（`docker ps` 可见 `Up`）。用 `docker exec -it yolo-dev bash` 进入即可。

### Q3：`cd /workspace` 报 `No such file or directory`？
**A**：说明你还在宿主机，不在容器内。`/workspace` 是容器内路径。

### Q4：训练报 `unable to allocate shared memory`？
**A**：Docker 默认 `/dev/shm` 只有 64MB，PyTorch 多进程数据加载不够用。启动容器时加 `--shm-size=8g`。

### Q5：`run.ps1` 在 WSL 里报 `syntax error near unexpected token`？
**A**：`run.ps1` 是 PowerShell 脚本，不能在 bash 里运行。用 `./docker/run_wsl.sh`。

### Q6：数据加载时报找不到图片？
**A**：检查 `data.yaml` 的 `path`/`train` 是否正确，以及 `Train.txt` 中的 `data/` 前缀问题（见第二节）。推荐 `train: images/Train` 直接指向目录。

### Q7：mAP 虚高怎么办？
**A**：因为 val 用的训练集本身（无独立验证集）。正式评估应划分独立 val 集，或在 picture2 上标注后做验证。

### Q8：想要更好的效果？
**A**：换更大模型（`yolo11s-seg.pt` / `yolo11m-seg.pt`）、补足 48 张无标签图片的标注、增加数据量、数据增强参数微调。

---

## 九、本次训练结果汇总

| 项目 | 数值 |
|------|------|
| 模型 | YOLO11n-seg（114 层，283 万参数，9.6 GFLOPs） |
| 训练轮数 | 200/200（跑满，未触发早停） |
| 训练耗时 | ~645 分钟（约 10.8 小时） |
| GPU | NVIDIA GeForce RTX 5060 Ti（16GB，占用约 3.3G） |
| **Box mAP50** | **0.754** |
| Box mAP50-95 | 0.449 |
| **Mask mAP50** | **0.756** |
| Mask mAP50-95 | 0.396 |
| 推理速度 | ~17ms/张（640x640） |

> 注：val 为训练集占位，上述 mAP 为训练集上的评估值，参考意义大于绝对意义。
