# C++ inference

YOLOWorkbench builds two ONNX Runtime programs:

- yolo_cpp for object detection
- yolo_seg for instance segmentation

Build inside the project's Docker container:

    cd /workspace/cpp
    ./build.sh build

Run with an exported ONNX model and an input image available in the container:

    ./build.sh run det --model ../runs/detect/yolo11n_detect/weights/best.onnx --image /workspace/path/to/input.jpg --output samples/detection_result.jpg
    ./build.sh run seg --model ../runs/segment/segmentation/weights/best.onnx --image /workspace/path/to/input.jpg --output samples/segmentation_result.jpg

Use --help with either executable to see the supported options. For model training, validation, and export instructions, see the repository README.
