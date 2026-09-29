#ifndef YOLO_SEGMENTOR_H
#define YOLO_SEGMENTOR_H

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

// ============================================================
// YOLOSegmentor
//
// YOLO 实例分割检测器封装（ONNX Runtime CUDA + OpenCV）。
//
// 结构与 YOLODetector 保持一致，同样采用 PIMPL 设计：
//   - 头文件不暴露 onnxruntime 相关类型，实现全部藏在
//     src/YOLOSegmentor.cpp 的 struct Impl 中
//   - 对库使用者：只依赖 OpenCV 头文件，无需安装 ONNX Runtime
//     头文件，编译更快、依赖隔离更干净
//   - 便于未来以静态库 / 动态库形式发布（ABI 稳定）
//
// 支持 YOLO11-seg 等标准分割模型的 ONNX 导出格式：
//   output0: [1, 4 + num_classes + mask_channels, num_predictions]
//            前 4 行 xywh，之后 num_classes 行类别得分，
//            最后 mask_channels 行掩码系数
//   output1: [1, mask_channels, mask_h, mask_w]  原型掩码
//
// 用法：
//   YOLOSegmentor segmentor("/path/to/seg-model.onnx");
//   auto segs = segmentor.segment(image);
//   segmentor.draw(image, segs);   // 在原始图副本上叠加掩码/画框
// ============================================================

struct Segmentation
{
    int class_id;
    float confidence;
    cv::Rect box;    // 原图坐标
    cv::Mat mask;    // CV_8UC1，与原图同尺寸，0/255 二值掩码
};

class YOLOSegmentor
{
public:
    // --------------------------------------------------------
    // 构造 / 析构 / 移动
    // --------------------------------------------------------

    // 加载模型，可指定 CUDA 设备号
    YOLOSegmentor(
        const std::string& model_path,
        int device_id = 0);

    // PIMPL 需要：析构/移动必须在 .cpp 中定义（Impl 是完整类型）
    ~YOLOSegmentor();

    YOLOSegmentor(YOLOSegmentor&&) noexcept;
    YOLOSegmentor& operator=(YOLOSegmentor&&) noexcept;

    // 禁止拷贝（Impl 持有不可拷贝的 Ort::Session）
    YOLOSegmentor(const YOLOSegmentor&) = delete;
    YOLOSegmentor& operator=(const YOLOSegmentor&) = delete;

    // --------------------------------------------------------
    // 核心接口
    // --------------------------------------------------------

    // 对一张图做完整分割推理，返回过滤+NMS 后的实例（含二值掩码）
    std::vector<Segmentation> segment(const cv::Mat& image);

    // 在原图副本上叠加所有掩码并画检测框（返回新图）
    cv::Mat draw(
        const cv::Mat& image,
        const std::vector<Segmentation>& segmentations) const;

    // --------------------------------------------------------
    // 配置（可在 segment 前调整）
    // --------------------------------------------------------

    void setConfThreshold(float t);
    void setNmsThreshold(float t);

    float confThreshold() const;
    float nmsThreshold() const;

    // 类别名称（用于画图；默认空，缺省时显示 class_<id>）
    void setClassNames(const std::vector<std::string>& names);
    const std::vector<std::string>& classNames() const;

    // --------------------------------------------------------
    // 模型元信息
    // --------------------------------------------------------

    int inputHeight() const;
    int inputWidth() const;
    int numClasses() const;
    int maskChannels() const;   // 掩码系数个数（如 YOLO11-seg 为 32）
    int maskHeight() const;     // 原型掩码高（如 160）
    int maskWidth() const;      // 原型掩码宽（如 160）
    std::vector<int64_t> inputShape() const;

private:
    // --------------------------------------------------------
    // 私有工具（仅 segment() 内部使用）
    // --------------------------------------------------------

    // letterbox 预处理（输出 640x640 张量 + 记录缩放/填充）
    std::vector<float> preprocess(
        const cv::Mat& image,
        float& scale,
        int& pad_x,
        int& pad_y) const;

    // YOLO 分割输出后处理（检测框 NMS + 掩码重建）
    std::vector<Segmentation> postprocess(
        const float* output_data,
        int num_predictions,
        const float* proto_data,
        int mask_h,
        int mask_w,
        float scale,
        int pad_x,
        int pad_y,
        const cv::Size& original_size) const;

    // PIMPL：实现细节全部放在 Impl 中
    struct Impl;

    std::unique_ptr<Impl> impl_;
};

#endif // YOLO_SEGMENTOR_H
