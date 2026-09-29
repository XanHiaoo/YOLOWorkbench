# ==========================================
# YOLO Training + C++ ONNX Runtime Environment
#
# CUDA 12.8
# Ubuntu 24.04
# Python 3.12
# PyTorch CUDA 12.8
# Ultralytics
# OpenCV
# ONNX Runtime C++ GPU
#
# BuildKit cache optimized
# ==========================================

FROM nvidia/cuda:12.8.1-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# ----------------------------------------------------------
# 基础环境
# ----------------------------------------------------------
ENV TZ=Asia/Shanghai

# ----------------------------------------------------------
# Ubuntu 软件源
#
# Ubuntu 24.04 使用 /etc/apt/sources.list.d/ubuntu.sources
# ----------------------------------------------------------
RUN sed -i \
    's@http://archive.ubuntu.com/ubuntu/@https://mirrors.aliyun.com/ubuntu/@g; \
     s@http://security.ubuntu.com/ubuntu/@https://mirrors.aliyun.com/ubuntu/@g' \
    /etc/apt/sources.list.d/ubuntu.sources

# ----------------------------------------------------------
# 系统依赖
#
# BuildKit cache:
#   /var/cache/apt
#
# 即使这一层因为修改 Dockerfile 而重新执行，
# 已下载的 .deb 仍然可以从 cache 中复用。
# ----------------------------------------------------------
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    apt-get update -o Acquire::Retries=5 \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        git \
        wget \
        curl \
        vim \
        nano \
        cmake \
        build-essential \
        pkg-config \
        libopencv-dev \
        gdb \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------
# python 命令
# ----------------------------------------------------------
RUN ln -sf /usr/bin/python3 /usr/bin/python

# ----------------------------------------------------------
# pip
# ----------------------------------------------------------
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    python -m pip install \
        --upgrade pip \
        --break-system-packages \
        --ignore-installed

# ----------------------------------------------------------
# PyTorch CUDA 12.8
#
# 注意：
# PyTorch 继续使用官方 cu128 源
#
# BuildKit 会缓存下载的 wheel。
# ----------------------------------------------------------
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install \
        torch \
        torchvision \
        torchaudio \
        --index-url https://download.pytorch.org/whl/cu128 \
        --break-system-packages

# ----------------------------------------------------------
# Python / YOLO 软件
#
# 国内 PyPI 镜像
#
# 注意：
# 这里不要使用 --index-url cu128，
# 否则普通 Python 包无法正常从 PyPI 获取。
# ----------------------------------------------------------
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install \
        ultralytics \
        opencv-python-headless \
        numpy \
        pandas \
        matplotlib \
        pillow \
        tqdm \
        pyyaml \
        --index-url https://mirrors.aliyun.com/pypi/simple/ \
        --break-system-packages

# ----------------------------------------------------------
# ONNX Runtime
# ----------------------------------------------------------
ARG ONNXRUNTIME_VERSION=1.26.0

RUN --mount=type=cache,id=onnxruntime-cache,target=/tmp/onnx-cache \
    if [ ! -f "/tmp/onnx-cache/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz" ]; then \
        wget -q \
        "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz" \
        -O "/tmp/onnx-cache/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz"; \
    fi \
    && rm -rf /opt/onnxruntime \
    && tar -xzf \
        "/tmp/onnx-cache/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz" \
        -C /opt \
    && mv \
        "/opt/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}" \
        /opt/onnxruntime

# ----------------------------------------------------------
# CUDA / ONNX Runtime 动态库
# ----------------------------------------------------------
ENV LD_LIBRARY_PATH="/opt/onnxruntime/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.12/dist-packages/nvidia/curand/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# ----------------------------------------------------------
# 工作目录
# ----------------------------------------------------------
WORKDIR /workspace

# ----------------------------------------------------------
# 默认启动 bash
# ----------------------------------------------------------
CMD ["/bin/bash"]