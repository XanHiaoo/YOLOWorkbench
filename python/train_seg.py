import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a YOLO instance-segmentation model."
    )
    parser.add_argument("--data", required=True, help="Path to a YOLO segmentation dataset YAML file.")
    parser.add_argument("--model", default="yolo11n-seg.pt", help="Pretrained model name or local checkpoint path.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="GPU index or cpu.")
    parser.add_argument("--name", default="segmentation")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_path}. Provide a YOLO-format dataset with --data."
        )

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=50,
        device=args.device,
        workers=8,
        name=args.name,
    )


if __name__ == "__main__":
    main()
