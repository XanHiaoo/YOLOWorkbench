#!/usr/bin/env bash
# ============================================================
# YOLO C++ 一键编译 + 运行（在容器内 /workspace/cpp 下执行）
#
# 用法：
#   ./build.sh                        # Debug 配置 + 编译 + 运行检测 (yolo_cpp)
#   ./build.sh build                  # Debug 配置 + 编译
#   ./build.sh run det [参数...]      # 运行 yolo_cpp（可透传参数）
#   ./build.sh run seg [参数...]      # 运行 yolo_seg（可透传参数）
#   ./build.sh run [参数...]          # 同 run det（保持旧行为）
#   ./build.sh clean                  # 清理 build 目录
#   ./build.sh debug                  # 强制用 Debug 重新配置 + 编译（可断点调试）
#   ./build.sh release                # 强制用 Release 重新配置 + 编译（测速用）
#
# 依赖：
#   - cmake / g++（镜像内已含）
#   - ONNX Runtime 位于 /opt/onnxruntime（可用环境变量覆盖）
# ============================================================

set -euo pipefail

# ------------------------------------------------------------
# 配置
# ------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
ONNX_ROOT="${ONNXRUNTIME_ROOT:-/opt/onnxruntime}"
BUILD_TYPE="${CMAKE_BUILD_TYPE:-Debug}"

# ------------------------------------------------------------
# LD_LIBRARY_PATH 自动补充
#
# 容器里的 LD_LIBRARY_PATH 可能只包含部分路径（例如只含
# /usr/local/cuda/lib64），缺少 pip 安装的 nvidia/cublas 等目录，
# 导致运行时找不到 libcublasLt.so.12 等 CUDA 依赖库。
# 这里把缺失的路径追加进去（已存在的跳过，不重复）。
#
# 只在运行 yolo_cpp / yolo_seg 时生效，不影响编译。
# ------------------------------------------------------------
setup_ld_path() {
    local paths="${LD_LIBRARY_PATH:-}"
    local candidates=(
        "/opt/onnxruntime/lib"
        "/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib"
        "/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib"
        "/usr/local/lib/python3.12/dist-packages/nvidia/cufft/lib"
        "/usr/local/lib/python3.12/dist-packages/nvidia/curand/lib"
        "/usr/local/cuda/lib64"
    )

    local changed=0

    for p in "${candidates[@]}"; do
        if [ -d "$p" ]; then
            case ":$paths:" in
                *":$p:"*) ;;            # 已存在，跳过
                *)
                    paths="${paths:+$paths:}$p"
                    changed=1
                    ;;
            esac
        fi
    done

    if [ "$changed" = "1" ]; then
        export LD_LIBRARY_PATH="$paths"
        info "LD_LIBRARY_PATH 已补充缺失路径"
    fi
}

# 颜色输出
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ------------------------------------------------------------
# 参数分发
# ------------------------------------------------------------
ACTION="${1:-build}"

# 把 det / seg（或完整可执行名）映射到 build 目录里的程序名
resolve_program() {
    case "$1" in
        det|yolo_cpp) echo "yolo_cpp" ;;
        seg|yolo_seg) echo "yolo_seg" ;;
        *)            echo "" ;;
    esac
}

clean_build() {
    if [ -d "$BUILD_DIR" ]; then
        info "清理 build 目录: $BUILD_DIR"
        rm -rf "$BUILD_DIR"
    fi
}

configure() {
    if [ ! -d "$BUILD_DIR" ]; then
        info "配置 CMake (BUILD_TYPE=$BUILD_TYPE, ONNXRUNTIME_ROOT=$ONNX_ROOT)"
        cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
            -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
            -DONNXRUNTIME_ROOT="$ONNX_ROOT"
    else
        warn "build 目录已存在，跳过配置（如需重新配置请先 clean 或 debug/release）"
    fi
}

# 强制用指定构建类型重新配置 + 编译
rebuild_with_type() {
    local type="$1"
    BUILD_TYPE="$type"
    info "切换构建类型为 $type（强制重新配置）"
    clean_build
    configure
    build
}

build() {
    info "编译中..."
    cmake --build "$BUILD_DIR" -j"$(nproc)"
    info "编译完成: ${BUILD_DIR}/yolo_cpp 与 ${BUILD_DIR}/yolo_seg"
}

run() {
    local program="${1:-det}"
    shift || true

    local bin
    bin="$(resolve_program "$program")"

    if [ -z "$bin" ]; then
        error "未知程序: $program（可用 det / seg）"
        exit 1
    fi

    if [ ! -f "$BUILD_DIR/$bin" ]; then
        error "未找到可执行文件 ${BUILD_DIR}/$bin，请先编译"
        exit 1
    fi

    setup_ld_path
    info "运行 $bin ..."
    echo "----------------------------------------"
    "$BUILD_DIR/$bin" "$@"
}

case "$ACTION" in
    all)     configure; build; run det ;;
    build)   configure; build ;;
    run)     shift; run "$@" ;;
    clean)   clean_build ;;
    debug)   rebuild_with_type "Debug" ;;
    release) rebuild_with_type "Release" ;;
    *)
        error "未知参数: $ACTION"
        echo "可用: all / build / run [det|seg] / clean / debug / release"
        exit 1
        ;;
esac

exit 0
