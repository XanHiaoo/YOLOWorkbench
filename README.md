# YOLOWorkbench

An end-to-end workspace for YOLO model training, validation, ONNX export, and inference with Python or C++.

## Quick start

The supported development environment is the Docker image defined in docker/Dockerfile. On a Windows PC with Docker Desktop, WSL2, and an NVIDIA GPU:

    git clone https://github.com/XanHiaoo/YOLOWorkbench.git
    cd YOLOWorkbench
    docker build -t yoloworkbench:cuda12.8 -f docker/Dockerfile .
    .\docker\run.ps1

The launcher mounts this checkout at /workspace in the container. Inside the container, build the C++ programs:

    cd /workspace/cpp
    ./build.sh build

Linux and WSL users can build the same image from the repository root, then start the container with:

    docker build -t yoloworkbench:cuda12.8 -f docker/Dockerfile .
    IMAGE=yoloworkbench:cuda12.8 ./docker/run_wsl.sh

C++ inference currently uses ONNX Runtime's CUDA provider and requires an NVIDIA GPU. The container can still configure and compile without passing a GPU; inference needs the GPU.

## Train and validate

Detection training uses Ultralytics' small COCO8 example dataset and downloads the standard YOLO11n pretrained weights when needed:

    python /workspace/python/train_detect.py
    yolo detect val model=runs/detect/yolo11n_detect/weights/best.pt data=coco8.yaml

Instance segmentation training needs a YOLO-format dataset. Keep the dataset outside Git. Copy configs/segmentation_dataset.example.yaml to the dataset root as data.yaml, then update its path and class names:



    python /workspace/python/train_seg.py --data /workspace/datasets/my-segmentation-dataset/data.yaml
    yolo segment val model=runs/segment/segmentation/weights/best.pt data=/workspace/datasets/my-segmentation-dataset/data.yaml

The YAML must reference the training and validation images and labels in YOLO format. Use --model to select a different pretrained or local model.

## Export and run inference

Export the trained weights to ONNX:

    yolo export model=runs/detect/yolo11n_detect/weights/best.pt format=onnx imgsz=640
    yolo export model=runs/segment/segmentation/weights/best.pt format=onnx imgsz=640

Python detection inference:

    python /workspace/python/onnx_inference.py --model /workspace/runs/detect/yolo11n_detect/weights/best.onnx --image /workspace/path/to/input.jpg --output-dir /workspace/runs/inference

C++ inference (run from /workspace/cpp):

    ./build.sh run det --model ../runs/detect/yolo11n_detect/weights/best.onnx --image /workspace/path/to/input.jpg --output samples/detection_result.jpg
    ./build.sh run seg --model ../runs/segment/segmentation/weights/best.onnx --image /workspace/path/to/input.jpg --output samples/segmentation_result.jpg

Replace the image path with an image available inside the container. The C++ programs print detections or segmentations and timing statistics, then save an annotated image. The segmentation program can also save per-instance masks with --dump-masks DIR.

## What Git tracks

Source code, build configuration, and guides are tracked. Datasets, trained weights, ONNX exports, training runs, local sample images, and generated visualizations are intentionally ignored. They may be large or contain project-specific data. Supply your dataset and input images on each machine; Ultralytics downloads standard pretrained weights by model name when needed.

The detailed guides are in docs/. The Dockerfile is the source of truth for Python, CUDA, OpenCV, ONNX Runtime, and C++ build dependencies.
