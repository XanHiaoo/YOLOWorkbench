# ============================================================
# 目标检测训练脚本（YOLO11n，COCO 示例）
#
# 用法（在容器内，工作目录为 /workspace）：
#   python python/train_detect.py
#
# 说明：
#   - 数据集：coco8.yaml（Ultralytics 内置小样本，首次自动下载）
#     * 若要训练自己的检测数据集，将 data 改为你的 data.yaml
#   - 预训练权重：yolo11n.pt
#   - 输出：runs/detect/train/
# ============================================================
from ultralytics import YOLO


def main():
    model = YOLO("yolo11n.pt")

    model.train(
        data="coco8.yaml",  # 换成你自己的数据集 yaml 即可
        epochs=5,
        imgsz=640,
        batch=16,
        device=0,          # GPU 编号；无 GPU 改 "cpu"
        workers=4,
        name="yolo11n_detect",
    )


if __name__ == "__main__":
    main()
