#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include "YOLODetector.h"


// ============================================================
// Configuration
// ============================================================

static constexpr int WARMUP_COUNT = 10;
static constexpr int BENCHMARK_COUNT = 100;


// ============================================================
// 显示帮助
// ============================================================

static void printUsage(const char* prog)
{
    std::cout
        << "Usage: " << prog << " [options]\n"
        << "  -m, --model <path>    ONNX model path\n"
        << "  -i, --image <path>    Input image path\n"
        << "  -o, --output <path>   Output image path\n"
        << "  -d, --device <id>     CUDA device id (default 0)\n"
        << "  -c, --conf <float>    Confidence threshold (default 0.25)\n"
        << "  -n, --nms <float>     NMS IoU threshold (default 0.45)\n"
        << "  -w, --warmup <int>    Warmup iterations (default 10)\n"
        << "  -b, --bench <int>     Benchmark iterations (default 100)\n"
        << "  -h, --help            Show this help\n";
}


// ============================================================
// Main
// ============================================================

int main(int argc, char* argv[])
{
    // ========================================================
    // 配置（默认值 + 命令行覆盖）
    // ========================================================

    std::string model_path =
        "/workspace/runs/detect/yolo11n_detect/weights/best.onnx";

    std::string image_path =
        "";

    std::string output_image_path =
        "/workspace/cpp/samples/detection_result.jpg";

    int device_id = 0;

    float conf_threshold = 0.25f;
    float nms_threshold  = 0.45f;

    int warmup_count = WARMUP_COUNT;
    int bench_count  = BENCHMARK_COUNT;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        auto require_value = [&](const std::string& opt) -> bool {
            if (i + 1 >= argc)
            {
                std::cerr
                    << "ERROR: Missing value for "
                    << opt
                    << "\n";
                return false;
            }
            return true;
        };

        if (arg == "--model" || arg == "-m")
        {
            if (!require_value(arg)) return -1;
            model_path = argv[++i];
        }
        else if (arg == "--image" || arg == "-i")
        {
            if (!require_value(arg)) return -1;
            image_path = argv[++i];
        }
        else if (arg == "--output" || arg == "-o")
        {
            if (!require_value(arg)) return -1;
            output_image_path = argv[++i];
        }
        else if (arg == "--device" || arg == "-d")
        {
            if (!require_value(arg)) return -1;
            device_id = std::stoi(argv[++i]);
        }
        else if (arg == "--conf" || arg == "-c")
        {
            if (!require_value(arg)) return -1;
            conf_threshold = std::stof(argv[++i]);
        }
        else if (arg == "--nms" || arg == "-n")
        {
            if (!require_value(arg)) return -1;
            nms_threshold = std::stof(argv[++i]);
        }
        else if (arg == "--warmup" || arg == "-w")
        {
            if (!require_value(arg)) return -1;
            warmup_count = std::stoi(argv[++i]);
        }
        else if (arg == "--bench" || arg == "-b")
        {
            if (!require_value(arg)) return -1;
            bench_count = std::stoi(argv[++i]);
        }
        else if (arg == "--help" || arg == "-h")
        {
            printUsage(argv[0]);
            return 0;
        }
        else
        {
            std::cerr
                << "ERROR: Unknown argument: "
                << arg
                << "\n";
            return -1;
        }
    }


    if (image_path.empty())
    {
        std::cerr << "ERROR: Specify an input image with --image.\n";
        printUsage(argv[0]);
        return -1;
    }
    std::cout
        << "========================================\n"
        << " YOLO C++ ONNX Runtime CUDA\n"
        << "========================================\n\n";

    std::cout
        << "[Config]\n"
        << "  Model  : " << model_path << "\n"
        << "  Image  : " << image_path << "\n"
        << "  Output : " << output_image_path << "\n"
        << "  Device : " << device_id << "\n"
        << "  Conf   : " << conf_threshold << "\n"
        << "  NMS    : " << nms_threshold << "\n\n";


    // ========================================================
    // 1. 读图
    // ========================================================

    cv::Mat image =
        cv::imread(image_path, cv::IMREAD_COLOR);

    if (image.empty())
    {
        std::cerr
            << "ERROR: Failed to read image: "
            << image_path
            << "\n";
        return -1;
    }

    std::cout
        << "[Image]\n"
        << "  Width    : " << image.cols << "\n"
        << "  Height   : " << image.rows << "\n"
        << "  Channels : " << image.channels() << "\n\n";


    // ========================================================
    // 2. 加载模型
    // ========================================================

    std::cout
        << "[ONNX Runtime]\n"
        << "Loading model: " << model_path << "\n";

    auto load_start =
        std::chrono::high_resolution_clock::now();

    YOLODetector detector(model_path, device_id);

    detector.setConfThreshold(conf_threshold);
    detector.setNmsThreshold(nms_threshold);

    auto load_end =
        std::chrono::high_resolution_clock::now();

    double load_time =
        std::chrono::duration<double, std::milli>(
            load_end - load_start
        ).count();

    std::cout
        << "Model loaded successfully!\n"
        << "Load time: "
        << std::fixed << std::setprecision(3)
        << load_time << " ms\n\n";

    std::cout
        << "[Model IO]\n"
        << "  Input shape : ";
    for (auto dim : detector.inputShape())
    {
        std::cout << dim << " ";
    }
    std::cout << "\n\n";


    // ========================================================
    // 3. Warmup（同时让动态输出 shape 确定下来）
    // ========================================================

    std::cout
        << "[Warmup]\n"
        << "Running " << warmup_count
        << " warmup iterations...\n";

    for (int i = 0; i < warmup_count; ++i)
    {
        detector.detect(image);
    }

    std::cout << "Warmup finished.\n";

    std::cout
        << "  Num classes : " << detector.numClasses() << "\n\n";


    // ========================================================
    // 4. Benchmark
    // ========================================================

    std::vector<double> times;
    times.reserve(bench_count);

    std::cout
        << "[Benchmark]\n"
        << "Running " << bench_count
        << " iterations...\n";

    std::vector<Detection> last_detections;

    for (int i = 0; i < bench_count; ++i)
    {
        auto t0 =
            std::chrono::high_resolution_clock::now();

        last_detections = detector.detect(image);

        auto t1 =
            std::chrono::high_resolution_clock::now();

        double elapsed =
            std::chrono::duration<double, std::milli>(
                t1 - t0
            ).count();

        times.push_back(elapsed);
    }


    // ========================================================
    // 5. 打印检测结果 + 画图
    // ========================================================

    std::cout
        << "\n"
        << "========================================\n"
        << " Inference Result\n"
        << "========================================\n";

    std::cout
        << "Detections: " << last_detections.size() << "\n";

    for (size_t d = 0; d < last_detections.size(); ++d)
    {
        const auto& det = last_detections[d];

        std::cout
            << "  [" << d << "] "
            << det.class_id
            << "  conf="
            << std::fixed << std::setprecision(3)
            << det.confidence
            << "  box=("
            << det.box.x << ", "
            << det.box.y << ", "
            << det.box.width << ", "
            << det.box.height << ")\n";
    }

    cv::Mat result =
        detector.draw(image, last_detections);

    if (cv::imwrite(output_image_path, result))
    {
        std::cout
            << "\nResult image saved to: "
            << output_image_path << "\n";
    }
    else
    {
        std::cerr
            << "\nERROR: Failed to write: "
            << output_image_path << "\n";
    }


    // ========================================================
    // 6. 统计
    // ========================================================

    double sum = 0.0;

    double min_time =
        *std::min_element(times.begin(), times.end());

    double max_time =
        *std::max_element(times.begin(), times.end());

    for (double t : times) sum += t;

    double average = sum / times.size();

    std::vector<double> sorted_times = times;
    std::sort(sorted_times.begin(), sorted_times.end());

    double median =
        sorted_times[sorted_times.size() / 2];

    std::cout
        << "\n"
        << "========================================\n"
        << " Benchmark Result\n"
        << "========================================\n";

    std::cout
        << "Iterations : " << bench_count << "\n"
        << "Average    : " << std::fixed << std::setprecision(3)
        << average << " ms\n"
        << "Median     : " << median << " ms\n"
        << "Min        : " << min_time << " ms\n"
        << "Max        : " << max_time << " ms\n";

    if (average > 0.0)
    {
        std::cout << "FPS        : " << 1000.0 / average << "\n";
    }

    std::cout
        << "\n"
        << "========================================\n"
        << " INFERENCE SUCCESS\n"
        << "========================================\n";

    return 0;
}
