#ifndef YOLO_DETECTOR_H
#define YOLO_DETECTOR_H

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

// ============================================================
// YOLODetector
//
// YOLO 目标检测器封装（ONNX Runtime CUDA + OpenCV）
//
// 采用 PIMPL 设计（Pointer to Implementation）：
//   - 头文件不暴露 onnxruntime 相关类型，实现全部藏在
//     src/YOLODetector.cpp 的 struct Impl 中
//   - 对库使用者：只依赖 OpenCV 头文件，无需安装 ONNX Runtime
//     的头文件，编译更快、依赖隔离更干净
//   - 便于未来以静态库 / 动态库形式发布（ABI 稳定）
//
// 用法：
//   YOLODetector detector("/path/to/model.onnx");
//   auto dets = detector.detect(image);
//   detector.draw(image, dets);   // 在原图/副本上画框
// ============================================================

struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};

class YOLODetector
{
public:
    // --------------------------------------------------------
    // 构造 / 析构 / 移动
    // --------------------------------------------------------

    // 加载模型，可指定 CUDA 设备号
    YOLODetector(
        const std::string& model_path,
        int device_id = 0);

    // PIMPL 需要：析构/移动必须在 .cpp 中定义（Impl 是完整类型）
    ~YOLODetector();

    YOLODetector(YOLODetector&&) noexcept;
    YOLODetector& operator=(YOLODetector&&) noexcept;

    // 禁止拷贝（Impl 持有不可拷贝的 Ort::Session）
    YOLODetector(const YOLODetector&) = delete;
    YOLODetector& operator=(const YOLODetector&) = delete;

    // --------------------------------------------------------
    // 核心接口
    // --------------------------------------------------------

    // 对一张图做完整检测，返回过滤+NMS 后的结果
    std::vector<Detection> detect(const cv::Mat& image);

    // 在原图副本上画出所有检测框（返回新图）
    cv::Mat draw(
        const cv::Mat& image,
        const std::vector<Detection>& detections) const;

    // --------------------------------------------------------
    // 配置（可在 detect 前调整）
    // --------------------------------------------------------

    void setConfThreshold(float t);
    void setNmsThreshold(float t);

    float confThreshold() const;
    float nmsThreshold() const;

    // 模型元信息
    int inputHeight() const;
    int inputWidth() const;
    int numClasses() const;
    std::vector<int64_t> inputShape() const;

private:
    // --------------------------------------------------------
    // 私有工具（仅 detect() 内部使用）
    // --------------------------------------------------------

    // letterbox 预处理（输出 640x640 张量 + 记录缩放/填充）
    std::vector<float> preprocess(
        const cv::Mat& image,
        float& scale,
        int& pad_x,
        int& pad_y) const;

    // YOLO 输出后处理
    std::vector<Detection> postprocess(
        const float* output,
        int num_predictions,
        float scale,
        int pad_x,
        int pad_y,
        const cv::Size& original_size) const;

    // PIMPL：实现细节全部放在 Impl 中
    struct Impl;

    std::unique_ptr<Impl> impl_;
};

#endif // YOLO_DETECTOR_H
