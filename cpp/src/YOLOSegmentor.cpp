#include "YOLOSegmentor.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <onnxruntime_cxx_api.h>

// ============================================================
// 默认类别名称（本项目分割模型为裂缝 crack）
// 如果你的模型不是这个，请在构造后调用 setClassNames() 覆盖
// ============================================================

static const std::vector<std::string> DEFAULT_CLASS_NAMES =
{
    "crack"
};

// ============================================================
// Impl：PIMPL 实现类
//
// 所有 ONNX Runtime 相关类型都藏在这里，头文件完全不可见。
// ============================================================

struct YOLOSegmentor::Impl
{
    // --------------------------------------------------------
    // 构造：加载模型 + 读取输入输出信息
    // --------------------------------------------------------

    Impl(const std::string& model_path, int device_id)
        : env(ORT_LOGGING_LEVEL_WARNING, "YOLOSegmentor")
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

        input_name = input_name_allocated.get();

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

        // 分割模型需要两个输出：
        //   output0: [1, 4 + num_classes + mask_channels, num_predictions]
        //   output1: [1, mask_channels, mask_h, mask_w]
        if (session.GetOutputCount() < 2)
        {
            throw std::runtime_error(
                "YOLOSegmentor: model must have 2 outputs "
                "(detection + prototype masks)"
            );
        }

        auto output0_name_allocated =
            session.GetOutputNameAllocated(0, allocator);

        auto output1_name_allocated =
            session.GetOutputNameAllocated(1, allocator);

        output0_name = output0_name_allocated.get();
        output1_name = output1_name_allocated.get();

        auto output0_info =
            session.GetOutputTypeInfo(0)
                .GetTensorTypeAndShapeInfo();

        auto output1_info =
            session.GetOutputTypeInfo(1)
                .GetTensorTypeAndShapeInfo();

        auto output0_shape = output0_info.GetShape();
        auto output1_shape = output1_info.GetShape();

        // 动态维度时此处可能是 -1，由 segment 首次推理后修正
        if (output0_shape.size() >= 2)
        {
            detection_channels =
                static_cast<int>(output0_shape[1]);
        }

        if (output1_shape.size() >= 3)
        {
            if (output1_shape[1] > 0)
            {
                mask_channels = static_cast<int>(output1_shape[1]);
            }
            if (output1_shape[2] > 0)
            {
                mask_h = static_cast<int>(output1_shape[2]);
            }
            if (output1_shape[3] > 0)
            {
                mask_w = static_cast<int>(output1_shape[3]);
            }
        }

        if (detection_channels > 4 && mask_channels > 0)
        {
            num_classes = detection_channels - 4 - mask_channels;
        }
    }

    // --------------------------------------------------------
    // ONNX Runtime 相关成员
    // --------------------------------------------------------

    Ort::Env env;
    Ort::Session session{nullptr};

    std::string input_name;
    std::string output0_name;
    std::string output1_name;
    std::vector<int64_t> input_shape;

    int input_w = 640;
    int input_h = 640;
    int detection_channels = 0;
    int num_classes = 0;
    int mask_channels = 32;
    int mask_h = 160;
    int mask_w = 160;

    float conf_threshold = 0.25f;
    float nms_threshold  = 0.45f;

    std::vector<std::string> class_names = DEFAULT_CLASS_NAMES;
};

// ============================================================
// 构造 / 析构 / 移动
// ============================================================

YOLOSegmentor::YOLOSegmentor(
    const std::string& model_path,
    int device_id)
    : impl_(std::make_unique<Impl>(model_path, device_id))
{
}

YOLOSegmentor::~YOLOSegmentor() = default;

YOLOSegmentor::YOLOSegmentor(YOLOSegmentor&&) noexcept = default;
YOLOSegmentor& YOLOSegmentor::operator=(YOLOSegmentor&&) noexcept = default;

// ============================================================
// 核心接口：委托给 impl_
// ============================================================

std::vector<Segmentation> YOLOSegmentor::segment(
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
    const char* output_names[] = { d.output0_name.c_str(), d.output1_name.c_str() };

    auto outputs =
        d.session.Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            2
        );

    // --------------------------------------------------------
    // 4. 读取输出并修正动态形状信息
    // --------------------------------------------------------

    const float* output_data =
        outputs[0].GetTensorData<float>();

    auto output0_shape =
        outputs[0]
            .GetTensorTypeAndShapeInfo()
            .GetShape();

    const float* proto_data =
        outputs[1].GetTensorData<float>();

    auto output1_shape =
        outputs[1]
            .GetTensorTypeAndShapeInfo()
            .GetShape();

    int num_predictions = 0;
    int mask_h = d.mask_h;
    int mask_w = d.mask_w;

    if (output0_shape.size() >= 3)
    {
        int channels = static_cast<int>(output0_shape[1]);
        num_predictions = static_cast<int>(output0_shape[2]);

        if (channels > 4)
        {
            d.detection_channels = channels;

            if (d.mask_channels > 0)
            {
                d.num_classes = channels - 4 - d.mask_channels;
            }
        }
    }

    if (output1_shape.size() >= 3)
    {
        if (output1_shape[1] > 0)
        {
            d.mask_channels = static_cast<int>(output1_shape[1]);
        }
        if (output1_shape[2] > 0)
        {
            d.mask_h = static_cast<int>(output1_shape[2]);
        }
        if (output1_shape[3] > 0)
        {
            d.mask_w = static_cast<int>(output1_shape[3]);
        }

        mask_h = d.mask_h;
        mask_w = d.mask_w;
    }

    // 动态维度场景：首次推理后重新推导类别数
    if (d.num_classes <= 0 &&
        d.detection_channels > 4 &&
        d.mask_channels > 0)
    {
        d.num_classes =
            d.detection_channels - 4 - d.mask_channels;
    }

    return postprocess(
        output_data,
        num_predictions,
        proto_data,
        mask_h,
        mask_w,
        scale,
        pad_x,
        pad_y,
        image.size()
    );
}

// ============================================================
// Draw：在原图副本上叠加掩码 + 画框 + 标签
// ============================================================

cv::Mat YOLOSegmentor::draw(
    const cv::Mat& image,
    const std::vector<Segmentation>& segmentations) const
{
    const Impl& d = *impl_;

    cv::Mat result = image.clone();

    // --------------------------------------------------------
    // 1. 先叠加所有掩码（半透明）
    // --------------------------------------------------------

    cv::Mat overlay = image.clone();

    for (const auto& seg : segmentations)
    {
        if (seg.mask.empty() ||
            seg.mask.size() != image.size())
        {
            continue;
        }

        // 按类别取颜色（简单调色板，超界回退绿色）
        const std::vector<cv::Scalar> palette =
        {
            cv::Scalar(0, 255, 0),    // green
            cv::Scalar(255, 0, 0),    // blue
            cv::Scalar(0, 0, 255),    // red
            cv::Scalar(255, 255, 0),  // cyan
            cv::Scalar(255, 0, 255),  // magenta
            cv::Scalar(0, 255, 255),  // yellow
        };

        cv::Scalar color = cv::Scalar(0, 255, 0);

        if (seg.class_id >= 0 &&
            seg.class_id < static_cast<int>(palette.size()))
        {
            color = palette[seg.class_id];
        }

        overlay.setTo(color, seg.mask);
    }

    cv::addWeighted(overlay, 0.5, result, 0.5, 0.0, result);

    // --------------------------------------------------------
    // 2. 画检测框 + 标签
    // --------------------------------------------------------

    for (const auto& seg : segmentations)
    {
        const cv::Rect& box = seg.box;

        cv::Scalar color = cv::Scalar(0, 255, 0);

        const std::vector<cv::Scalar> palette =
        {
            cv::Scalar(0, 255, 0),
            cv::Scalar(255, 0, 0),
            cv::Scalar(0, 0, 255),
            cv::Scalar(255, 255, 0),
            cv::Scalar(255, 0, 255),
            cv::Scalar(0, 255, 255),
        };

        if (seg.class_id >= 0 &&
            seg.class_id < static_cast<int>(palette.size()))
        {
            color = palette[seg.class_id];
        }

        cv::rectangle(result, box, color, 2);

        std::string class_name;

        if (seg.class_id >= 0 &&
            seg.class_id < static_cast<int>(d.class_names.size()))
        {
            class_name = d.class_names[seg.class_id];
        }
        else
        {
            class_name = "class_" + std::to_string(seg.class_id);
        }

        std::ostringstream label;
        label << class_name
              << " "
              << std::fixed
              << std::setprecision(2)
              << seg.confidence;

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
            color,
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

void YOLOSegmentor::setConfThreshold(float t) { impl_->conf_threshold = t; }
void YOLOSegmentor::setNmsThreshold(float t)  { impl_->nms_threshold  = t; }

float YOLOSegmentor::confThreshold() const { return impl_->conf_threshold; }
float YOLOSegmentor::nmsThreshold()  const { return impl_->nms_threshold;  }

void YOLOSegmentor::setClassNames(
    const std::vector<std::string>& names)
{
    impl_->class_names = names;
}

const std::vector<std::string>& YOLOSegmentor::classNames() const
{
    return impl_->class_names;
}

int YOLOSegmentor::inputHeight() const { return impl_->input_h; }
int YOLOSegmentor::inputWidth()  const { return impl_->input_w; }
int YOLOSegmentor::numClasses()  const { return impl_->num_classes; }
int YOLOSegmentor::maskChannels() const { return impl_->mask_channels; }
int YOLOSegmentor::maskHeight()  const { return impl_->mask_h; }
int YOLOSegmentor::maskWidth()   const { return impl_->mask_w; }

std::vector<int64_t> YOLOSegmentor::inputShape() const
{
    return impl_->input_shape;
}

// ============================================================
// Preprocess：letterbox -> RGB -> float -> HWC->CHW
// ============================================================

std::vector<float> YOLOSegmentor::preprocess(
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
// Postprocess：置信度过滤 -> 坐标还原 -> NMS -> 掩码重建
// ============================================================

std::vector<Segmentation> YOLOSegmentor::postprocess(
    const float* output_data,
    int num_predictions,
    const float* proto_data,
    int mask_h,
    int mask_w,
    float scale,
    int pad_x,
    int pad_y,
    const cv::Size& original_size) const
{
    const Impl& d = *impl_;

    // --------------------------------------------------------
    // 1. 置信度过滤（letterboxed 空间，640x640）
    // --------------------------------------------------------

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<int> cand_indices;

    boxes.reserve(num_predictions);
    scores.reserve(num_predictions);
    class_ids.reserve(num_predictions);
    cand_indices.reserve(num_predictions);

    const int canvas_w = d.input_w;
    const int canvas_h = d.input_h;

    for (int i = 0; i < num_predictions; ++i)
    {
        float cx = output_data[0 * num_predictions + i];
        float cy = output_data[1 * num_predictions + i];
        float w  = output_data[2 * num_predictions + i];
        float h  = output_data[3 * num_predictions + i];

        // 找最佳类别
        int best_class = -1;
        float best_score = 0.0f;

        for (int c = 0; c < d.num_classes; ++c)
        {
            float score =
                output_data[(4 + c) * num_predictions + i];

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

        // 裁剪到 letterboxed 画布边界
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(canvas_w - 1)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(canvas_h - 1)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(canvas_w - 1)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(canvas_h - 1)));

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
        cand_indices.push_back(i);
    }

    // --------------------------------------------------------
    // 2. NMS
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
    // 3. 掩码重建
    // --------------------------------------------------------

    // 原型掩码按 CHW 布局：proto[k][y * mask_w + x]
    // 系数位于 output_data 的 (4 + num_classes + k) 行
    const int proto_plane = mask_h * mask_w;

    std::vector<Segmentation> segmentations;

    for (int index : indices)
    {
        int cand = cand_indices[index];

        // 3.1 组合原型掩码 -> 160x160 logits
        cv::Mat logits(mask_h, mask_w, CV_32F);

        for (int y = 0; y < mask_h; ++y)
        {
            float* row = logits.ptr<float>(y);

            for (int x = 0; x < mask_w; ++x)
            {
                float acc = 0.0f;

                for (int k = 0; k < d.mask_channels; ++k)
                {
                    float coeff =
                        output_data[
                            (4 + d.num_classes + k) *
                            num_predictions + cand
                        ];

                    acc +=
                        coeff *
                        proto_data[k * proto_plane + y * mask_w + x];
                }

                row[x] = acc;
            }
        }

        // 3.2 双线性上采样到 640x640，阈值 > 0（等价于 sigmoid > 0.5）
        cv::Mat logits640;
        cv::resize(
            logits,
            logits640,
            cv::Size(canvas_w, canvas_h),
            0.0,
            0.0,
            cv::INTER_LINEAR
        );

        cv::Mat mask640 = cv::Mat::zeros(
            canvas_h, canvas_w, CV_8UC1
        );

        for (int y = 0; y < canvas_h; ++y)
        {
            const float* src_row = logits640.ptr<float>(y);
            uchar* dst_row = mask640.ptr<uchar>(y);

            for (int x = 0; x < canvas_w; ++x)
            {
                dst_row[x] = src_row[x] > 0.0f ? 255 : 0;
            }
        }

        // 3.3 只保留检测框内的掩码（letterboxed 空间）
        const cv::Rect& box640 = boxes[index];

        cv::Mat mask_box = cv::Mat::zeros(
            canvas_h, canvas_w, CV_8UC1
        );

        mask640(box640).copyTo(mask_box(box640));

        // 3.4 去掉 letterbox padding，缩放到原图尺寸
        int new_w = static_cast<int>(std::round(original_size.width * scale));
        int new_h = static_cast<int>(std::round(original_size.height * scale));

        cv::Rect content_rect(
            pad_x,
            pad_y,
            new_w,
            new_h
        );

        cv::Mat content =
            mask_box(content_rect).clone();

        // 转成 float32（0-255）再做双线性缩放，避免 uint8 精度损失
        cv::Mat content_f;
        content.convertTo(content_f, CV_32F);

        cv::Mat soft;
        cv::resize(
            content_f,
            soft,
            original_size,
            0.0,
            0.0,
            cv::INTER_LINEAR
        );

        cv::Mat mask_orig = cv::Mat::zeros(
            original_size.height, original_size.width, CV_8UC1
        );

        for (int y = 0; y < original_size.height; ++y)
        {
            const float* src_row = soft.ptr<float>(y);
            uchar* dst_row = mask_orig.ptr<uchar>(y);

            for (int x = 0; x < original_size.width; ++x)
            {
                // 阈值 127.5 等价于归一化 [0,1] 尺度下的 0.5
                dst_row[x] = src_row[x] > 127.5f ? 255 : 0;
            }
        }

        // 3.5 检测框还原到原图坐标
        float ox1 = (box640.x - pad_x) / scale;
        float oy1 = (box640.y - pad_y) / scale;
        float ox2 = (box640.x + box640.width - pad_x) / scale;
        float oy2 = (box640.y + box640.height - pad_y) / scale;

        ox1 = std::max(0.0f, std::min(ox1, static_cast<float>(original_size.width  - 1)));
        oy1 = std::max(0.0f, std::min(oy1, static_cast<float>(original_size.height - 1)));
        ox2 = std::max(0.0f, std::min(ox2, static_cast<float>(original_size.width  - 1)));
        oy2 = std::max(0.0f, std::min(oy2, static_cast<float>(original_size.height - 1)));

        int o_width  = static_cast<int>(ox2 - ox1);
        int o_height = static_cast<int>(oy2 - oy1);

        if (o_width <= 1 || o_height <= 1)
        {
            continue;
        }

        Segmentation seg;

        seg.class_id   = class_ids[index];
        seg.confidence = scores[index];
        seg.box        = cv::Rect(
            static_cast<int>(ox1),
            static_cast<int>(oy1),
            o_width,
            o_height
        );
        seg.mask       = mask_orig;

        segmentations.push_back(seg);
    }

    return segmentations;
}
