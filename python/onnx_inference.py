import os
import time
import argparse
import cv2
import numpy as np
import onnxruntime as ort

# ============================================================
# 参数化配置（默认值保留原来的 /workspace 路径）
#
# 用法示例：
#   python onnx_inference.py \
#       -m /workspace/models/best.onnx \
#       -i /workspace/cpp/samples/test.jpg \
#       -o /workspace/runs/detect/onnx
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO ONNX Runtime 推理（CUDA）",
    )
    parser.add_argument(
        "-m", "--model",
        required=True,
        help="ONNX 模型路径",
    )
    parser.add_argument(
        "-i", "--image",
        required=True,
        help="输入图片路径",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="runs/inference",
        help="输出目录（结果保存为 <output_dir>/result.jpg）",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="输入尺寸（默认 640）",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="置信度阈值（默认 0.25）",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU 阈值（默认 0.45）",
    )
    return parser.parse_args()


NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
    50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
    54: "donut", 55: "cake", 56: "chair", 57: "couch",
    58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet",
    62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard",
    67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster",
    71: "sink", 72: "refrigerator", 73: "book", 74: "clock", 75: "vase",
    76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush",
}


def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = image.shape[:2]
    new_h, new_w = new_shape

    ratio = min(new_w / w, new_h / h)

    resized_w = int(round(w * ratio))
    resized_h = int(round(h * ratio))

    if (w, h) != (resized_w, resized_h):
        image = cv2.resize(
            image,
            (resized_w, resized_h),
            interpolation=cv2.INTER_LINEAR,
        )

    dw = new_w - resized_w
    dh = new_h - resized_h

    dw /= 2
    dh /= 2

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color,
    )

    return image, ratio, (dw, dh)


def preprocess(image):
    image, ratio, pad = letterbox(
        image,
        (IMG_SIZE, IMG_SIZE),
    )

    # BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # HWC -> CHW
    image = image.transpose(2, 0, 1)

    # uint8 -> float32 -> [0, 1]
    image = np.ascontiguousarray(
        image,
        dtype=np.float32,
    ) / 255.0

    # [3,640,640] -> [1,3,640,640]
    image = np.expand_dims(image, axis=0)

    return image, ratio, pad


def postprocess(output, original_shape, ratio, pad):
    """
    YOLO output:
        [1, 84, 8400]

    Convert to:
        [8400, 84]

    Each candidate:
        [x, y, w, h, class0, ..., class79]
    """

    predictions = output[0].transpose(1, 0)

    boxes = predictions[:, :4]
    class_scores = predictions[:, 4:]

    class_ids = np.argmax(
        class_scores,
        axis=1,
    )

    scores = class_scores[
        np.arange(len(class_scores)),
        class_ids,
    ]

    # Confidence filtering
    mask = scores >= CONF_THRES

    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return []

    # xywh -> xyxy
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2

    # Remove letterbox padding
    dw, dh = pad

    x1 = (x1 - dw) / ratio
    y1 = (y1 - dh) / ratio
    x2 = (x2 - dw) / ratio
    y2 = (y2 - dh) / ratio

    orig_h, orig_w = original_shape[:2]

    x1 = np.clip(x1, 0, orig_w - 1)
    y1 = np.clip(y1, 0, orig_h - 1)
    x2 = np.clip(x2, 0, orig_w - 1)
    y2 = np.clip(y2, 0, orig_h - 1)

    boxes_xyxy = np.stack(
        [x1, y1, x2, y2],
        axis=1,
    )

    # Class-aware NMS
    final_indices = []

    for cls in np.unique(class_ids):

        cls_indices = np.where(
            class_ids == cls
        )[0]

        cls_boxes = boxes_xyxy[cls_indices]
        cls_scores = scores[cls_indices]

        nms_boxes = []

        for box in cls_boxes:
            nms_boxes.append([
                float(box[0]),
                float(box[1]),
                float(box[2] - box[0]),
                float(box[3] - box[1]),
            ])

        keep = cv2.dnn.NMSBoxes(
            nms_boxes,
            cls_scores.tolist(),
            CONF_THRES,
            IOU_THRES,
        )

        if len(keep) > 0:

            keep = np.array(
                keep
            ).reshape(-1)

            final_indices.extend(
                cls_indices[keep].tolist()
            )

    results = []

    for idx in final_indices:

        class_id = int(class_ids[idx])

        results.append({
            "box": boxes_xyxy[idx].tolist(),
            "score": float(scores[idx]),
            "class_id": class_id,
            "class_name": NAMES.get(
                class_id,
                str(class_id),
            ),
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return results


def main():

    args = parse_args()

    MODEL_PATH = args.model
    IMAGE_PATH = args.image
    OUTPUT_DIR = args.output_dir
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, "result.jpg")

    global IMG_SIZE, CONF_THRES, IOU_THRES
    IMG_SIZE = args.img_size
    CONF_THRES = args.conf
    IOU_THRES = args.iou

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    print("================================")
    print("YOLO ONNX Inference")
    print("================================")

    print("Model:", MODEL_PATH)
    print("Image:", IMAGE_PATH)

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            MODEL_PATH
        )

    if not os.path.isfile(IMAGE_PATH):
        raise FileNotFoundError(
            IMAGE_PATH
        )

    # Load CUDA/cuDNN DLLs from PyTorch
    ort.preload_dlls()

    session = ort.InferenceSession(
        MODEL_PATH,
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    print(
        "Providers:",
        session.get_providers(),
    )

    input_info = session.get_inputs()[0]

    print(
        "Input:",
        input_info.name,
        input_info.shape,
        input_info.type,
    )

    print(
        "Output:",
        session.get_outputs()[0].name,
        session.get_outputs()[0].shape,
    )

    # Read image
    image = cv2.imread(
        IMAGE_PATH
    )

    if image is None:
        raise RuntimeError(
            "Failed to read image"
        )

    print(
        "Original image:",
        image.shape,
    )

    # Preprocess
    input_tensor, ratio, pad = preprocess(
        image
    )

    print(
        "Input tensor:",
        input_tensor.shape,
        input_tensor.dtype,
    )

    print(
        "Letterbox ratio:",
        ratio,
    )

    print(
        "Letterbox pad:",
        pad,
    )

    # ============================
    # Inference
    # ============================

    print("Running inference...")

    t0 = time.perf_counter()

    outputs = session.run(
        None,
        {
            input_info.name:
            input_tensor
        },
    )

    inference_ms = (
        time.perf_counter() - t0
    ) * 1000

    print(
        "Inference:",
        f"{inference_ms:.3f} ms",
    )

    print(
        "Raw output:",
        outputs[0].shape,
    )

    # ============================
    # Postprocess
    # ============================

    t0 = time.perf_counter()

    detections = postprocess(
        outputs[0],
        image.shape,
        ratio,
        pad,
    )

    postprocess_ms = (
        time.perf_counter() - t0
    ) * 1000

    print(
        "Postprocess:",
        f"{postprocess_ms:.3f} ms",
    )

    print(
        "Detections:",
        len(detections),
    )

    # ============================
    # Draw
    # ============================

    result = image.copy()

    for det in detections:

        x1, y1, x2, y2 = [
            int(round(v))
            for v in det["box"]
        ]

        score = det["score"]
        class_name = det["class_name"]

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        label = (
            f"{class_name} "
            f"{score:.2f}"
        )

        cv2.putText(
            result,
            label,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        print(
            f"  {class_name:<15} "
            f"score={score:.4f} "
            f"box=({x1},{y1},{x2},{y2})"
        )

    cv2.imwrite(
        OUTPUT_PATH,
        result,
    )

    print()
    print(
        "Result saved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
