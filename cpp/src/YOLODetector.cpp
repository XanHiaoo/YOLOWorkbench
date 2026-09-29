#include "YOLODetector.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <onnxruntime_cxx_api.h>

// ============================================================
// COCO 类别名（80 类）
// 如果你的模型不是 COCO，请修改这里
// ============================================================

static const std::vector<std::string> CLASS_NAMES =
{
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

// ============================================================
// Impl：PIMPL 实现类
//
// 所有 ONNX Runtime 相关类型都藏在这里，头文件完全不可见。
// ============================================================

struct YOLODetector::Impl
{
    // --------------------------------------------------------
    // 构造：加载模型 + 读取输入输出信息
    // --------------------------------------------------------

    Impl(const std::string& model_path, int device_id)
        : env(ORT_LOGGING_LEVEL_WARNING, "YOLO")
    {
        Ort::SessionOptions session_options;

        session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL
        );

        OrtCUDAProviderOptions cuda_options{};

        cuda_options.device_id = device_id;

        session_options.AppendExecutionProvider_CUDA(
            cuda_options
        );

        // 加载模型
        session = Ort::Session(
            env,
            model_path.c_str(),
            session_options
        );

        // 读取输入 / 输出名字和形状
        Ort::AllocatorWithDefaultOptions allocator;

        auto input_name_allocated =
            session.GetInputNameAllocated(0, allocator);

        auto output_name_allocated =
            session.GetOutputNameAllocated(0, allocator);

        input_name  = input_name_allocated.get();
        output_name = output_name_allocated.get();

        auto input_info =
            session.GetInputTypeInfo(0)
                .GetTensorTypeAndShapeInfo();

        input_shape = input_info.GetShape();

        // 从模型输入推导 H/W（动态维度时回退默认 640）
        if (input_shape.size() >= 4)
        {
            if (input_shape[2] > 0) input_h = static_cast<int>(input_shape[2]);
            if (input_shape[3] > 0) input_w = static_cast<int>(input_shape[3]);
        }

        auto output_info =
            session.GetOutputTypeInfo(0)
                .GetTensorTypeAndShapeInfo();

        auto output_shape = output_info.GetShape();

        // 输出 [1, 4+num_classes, num_predictions]
        // （动态维度时此处可能为 -1，由 detect 首次推理后修正）
        if (output_shape.size() >= 2)
        {
            int channels = static_cast<int>(output_shape[1]);
            if (channels > 4)
            {
                num_classes = channels - 4;
            }
        }
    }

    // --------------------------------------------------------
    // ONNX Runtime 相关成员
    // --------------------------------------------------------

    Ort::Env env;
    Ort::Session session{nullptr};

    std::string input_name;
    std::string output_name;
    std::vector<int64_t> input_shape;

    int input_w = 640;
    int input_h = 640;
    int num_classes = 0;

    float conf_threshold = 0.25f;
    float nms_threshold  = 0.45f;
};

// ============================================================
// 构造 / 析构 / 移动
// ============================================================

YOLODetector::YOLODetector(
    const std::string& model_path,
    int device_id)
    : impl_(std::make_unique<Impl>(model_path, device_id))
{
}

YOLODetector::~YOLODetector() = default;

YOLODetector::YOLODetector(YOLODetector&&) noexcept = default;
YOLODetector& YOLODetector::operator=(YOLODetector&&) noexcept = default;

// ============================================================
// 核心接口：委托给 impl_
// ============================================================

std::vector<Detection> YOLODetector::detect(
    const cv::Mat& image)
{
    Impl& d = *impl_;

    // --------------------------------------------------------
    // 1. 预处理
    // --------------------------------------------------------

    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;

    std::vector<float> input_tensor_values =
        preprocess(image, scale, pad_x, pad_y);

    // --------------------------------------------------------
    // 2. 构造输入张量
    // --------------------------------------------------------

    std::vector<int64_t> shape =
    {
        1, 3, d.input_h, d.input_w
    };

    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault
        );

    Ort::Value input_tensor =
        Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensor_values.data(),
            input_tensor_values.size(),
            shape.data(),
            shape.size()
        );

    // --------------------------------------------------------
    // 3. 推理
    // --------------------------------------------------------

    const char* input_names[]  = { d.input_name.c_str() };
    const char* output_names[] = { d.output_name.c_str() };

    auto outputs =
        d.session.Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            1
        );

    // --------------------------------------------------------
    // 4. 后处理
    // --------------------------------------------------------

    const float* output_data =
        outputs[0].GetTensorData<float>();

    auto output_shape =
        outputs[0]
            .GetTensorTypeAndShapeInfo()
            .GetShape();

    int num_predictions = 0;

    if (output_shape.size() >= 3)
    {
        int channels = static_cast<int>(output_shape[1]);
        num_predictions = static_cast<int>(output_shape[2]);

        // 模型输出为动态维度时，首次推理后用真实 shape 更新类别数
        if (d.num_classes <= 0 && channels > 4)
        {
            d.num_classes = channels - 4;
        }
    }

    return postprocess(
        output_data,
        num_predictions,
        scale,
        pad_x,
        pad_y,
        image.size()
    );
}

// ============================================================
// Draw：在副本上画框 + 标签
// ============================================================

cv::Mat YOLODetector::draw(
    const cv::Mat& image,
    const std::vector<Detection>& detections) const
{
    cv::Mat result = image.clone();

    for (const auto& det : detections)
    {
        const cv::Rect& box = det.box;

        cv::rectangle(result, box, cv::Scalar(0, 255, 0), 2);

        std::string class_name;

        if (det.class_id >= 0 &&
            det.class_id < static_cast<int>(CLASS_NAMES.size()))
        {
            class_name = CLASS_NAMES[det.class_id];
        }
        else
        {
            class_name = "class_" + std::to_string(det.class_id);
        }

        std::ostringstream label;
        label << class_name
              << " "
              << std::fixed
              << std::setprecision(2)
              << det.confidence;

        int baseline = 0;

        cv::Size text_size =
            cv::getTextSize(
                label.str(),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                1,
                &baseline
            );

        int x = std::max(0, box.x);
        int y = std::max(text_size.height, box.y);

        cv::rectangle(
            result,
            cv::Rect(
                x,
                y - text_size.height,
                text_size.width,
                text_size.height + baseline
            ),
            cv::Scalar(0, 255, 0),
            cv::FILLED
        );

        cv::putText(
            result,
            label.str(),
            cv::Point(x, y),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 0, 0),
            1
        );
    }

    return result;
}

// ============================================================
// 配置 getter/setter（委托给 impl_）
// ============================================================

void YOLODetector::setConfThreshold(float t) { impl_->conf_threshold = t; }
void YOLODetector::setNmsThreshold(float t)  { impl_->nms_threshold  = t; }

float YOLODetector::confThreshold() const { return impl_->conf_threshold; }
float YOLODetector::nmsThreshold()  const { return impl_->nms_threshold;  }

int YOLODetector::inputHeight() const { return impl_->input_h; }
int YOLODetector::inputWidth()  const { return impl_->input_w; }
int YOLODetector::numClasses()  const { return impl_->num_classes; }

std::vector<int64_t> YOLODetector::inputShape() const
{
    return impl_->input_shape;
}

// ============================================================
// Preprocess：letterbox -> RGB -> float -> HWC->CHW
// ============================================================

std::vector<float> YOLODetector::preprocess(
    const cv::Mat& image,
    float& scale,
    int& pad_x,
    int& pad_y) const
{
    const Impl& d = *impl_;

    const int original_w = image.cols;
    const int original_h = image.rows;

    scale = std::min(
        static_cast<float>(d.input_w) / original_w,
        static_cast<float>(d.input_h) / original_h
    );

    int new_w = static_cast<int>(std::round(original_w * scale));
    int new_h = static_cast<int>(std::round(original_h * scale));

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h));

    pad_x = (d.input_w - new_w) / 2;
    pad_y = (d.input_h - new_h) / 2;

    cv::Mat input;

    cv::copyMakeBorder(
        resized,
        input,
        pad_y,
        d.input_h - new_h - pad_y,
        pad_x,
        d.input_w - new_w - pad_x,
        cv::BORDER_CONSTANT,
        cv::Scalar(114, 114, 114)
    );

    // BGR -> RGB
    cv::cvtColor(input, input, cv::COLOR_BGR2RGB);

    // uint8 -> float32 [0,1]
    input.convertTo(input, CV_32F, 1.0 / 255.0);

    // HWC -> CHW
    std::vector<float> tensor(3 * d.input_h * d.input_w);

    std::vector<cv::Mat> channels;

    for (int c = 0; c < 3; ++c)
    {
        channels.emplace_back(
            d.input_h,
            d.input_w,
            CV_32F,
            tensor.data() + c * d.input_h * d.input_w
        );
    }

    cv::split(input, channels);

    return tensor;
}

// ============================================================
// Postprocess：置信度过滤 -> 坐标还原 -> NMS
// ============================================================

std::vector<Detection> YOLODetector::postprocess(
    const float* output,
    int num_predictions,
    float scale,
    int pad_x,
    int pad_y,
    const cv::Size& original_size) const
{
    const Impl& d = *impl_;

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;

    boxes.reserve(num_predictions);
    scores.reserve(num_predictions);
    class_ids.reserve(num_predictions);

    for (int i = 0; i < num_predictions; ++i)
    {
        float cx = output[0 * num_predictions + i];
        float cy = output[1 * num_predictions + i];
        float w  = output[2 * num_predictions + i];
        float h  = output[3 * num_predictions + i];

        // 找最佳类别
        int best_class = -1;
        float best_score = 0.0f;

        for (int c = 0; c < d.num_classes; ++c)
        {
            float score =
                output[(4 + c) * num_predictions + i];

            if (score > best_score)
            {
                best_score = score;
                best_class = c;
            }
        }

        // 置信度过滤
        if (best_score < d.conf_threshold)
        {
            continue;
        }

        // xywh -> xyxy
        float x1 = cx - w * 0.5f;
        float y1 = cy - h * 0.5f;
        float x2 = cx + w * 0.5f;
        float y2 = cy + h * 0.5f;

        // 去除 letterbox padding，还原到原图
        x1 = (x1 - pad_x) / scale;
        y1 = (y1 - pad_y) / scale;
        x2 = (x2 - pad_x) / scale;
        y2 = (y2 - pad_y) / scale;

        // 裁剪到原图边界
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(original_size.width  - 1)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(original_size.height - 1)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(original_size.width  - 1)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(original_size.height - 1)));

        int width  = static_cast<int>(x2 - x1);
        int height = static_cast<int>(y2 - y1);

        if (width <= 1 || height <= 1)
        {
            continue;
        }

        boxes.emplace_back(
            static_cast<int>(x1),
            static_cast<int>(y1),
            width,
            height
        );

        scores.push_back(best_score);
        class_ids.push_back(best_class);
    }

    // --------------------------------------------------------
    // NMS
    // --------------------------------------------------------

    std::vector<int> indices;

    cv::dnn::NMSBoxes(
        boxes,
        scores,
        d.conf_threshold,
        d.nms_threshold,
        indices
    );

    // --------------------------------------------------------
    // 组装结果
    // --------------------------------------------------------

    std::vector<Detection> detections;

    for (int index : indices)
    {
        Detection det;

        det.class_id   = class_ids[index];
        det.confidence = scores[index];
        det.box        = boxes[index];

        detections.push_back(det);
    }

    return detections;
}
