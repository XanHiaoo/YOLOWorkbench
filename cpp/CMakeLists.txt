cmake_minimum_required(VERSION 3.20)

project(YOLOWorkbench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================
# 构建类型
#
#   Debug   -> 含调试符号 (-g)，可断点调试（默认）
#   Release -> 优化 (-O3)，用于测速/发布
#
# 命令行切换：
#   cmake -DCMAKE_BUILD_TYPE=Release ..
#   cmake -DCMAKE_BUILD_TYPE=Debug ..
# ============================================================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

# ============================================================
# OpenCV
# ============================================================

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")


# ============================================================
# ONNX Runtime
#
# 默认安装路径：/opt/onnxruntime
#
# 可在命令行覆盖，例如：
#   cmake -DONNXRUNTIME_ROOT=/custom/path/onnxruntime ..
# ============================================================

set(ONNXRUNTIME_ROOT "/opt/onnxruntime"
    CACHE PATH "ONNX Runtime installation root directory"
)

set(ONNXRUNTIME_INCLUDE_DIR
    "${ONNXRUNTIME_ROOT}/include"
)

set(ONNXRUNTIME_LIB_DIR
    "${ONNXRUNTIME_ROOT}/lib"
)

message(STATUS "ONNX Runtime root: ${ONNXRUNTIME_ROOT}")
message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime lib: ${ONNXRUNTIME_LIB_DIR}")

# 官方预编译包的头文件是扁平结构：
#   include/onnxruntime_cxx_api.h
# 旧版（源码安装）头文件在：
#   include/onnxruntime/core/session/onnxruntime_cxx_api.h

set(ONNXRUNTIME_HAS_FLAT_HEADER FALSE)
set(ONNXRUNTIME_HAS_SOURCE_HEADER FALSE)

if(EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
    set(ONNXRUNTIME_HAS_FLAT_HEADER TRUE)
endif()

if(EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime/core/session/onnxruntime_cxx_api.h")
    set(ONNXRUNTIME_HAS_SOURCE_HEADER TRUE)
endif()

if(NOT ONNXRUNTIME_HAS_FLAT_HEADER AND NOT ONNXRUNTIME_HAS_SOURCE_HEADER)
    message(FATAL_ERROR
        "ONNX Runtime header not found: "
        "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h"
    )
endif()

# 旧版（源码安装）布局需要把 core/session 目录加入 include 路径，
# 这样 #include <onnxruntime_cxx_api.h> 才能找到头文件
set(ONNXRUNTIME_EXTRA_INCLUDE_DIR "")

if(ONNXRUNTIME_HAS_SOURCE_HEADER AND NOT ONNXRUNTIME_HAS_FLAT_HEADER)
    set(ONNXRUNTIME_EXTRA_INCLUDE_DIR
        "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime/core/session"
    )
endif()

if(NOT EXISTS "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so")
    message(FATAL_ERROR
        "ONNX Runtime library not found: "
        "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so"
    )
endif()


# ============================================================
# YOLODetector 库
#
# 发布形态：
#   - BUILD_SHARED_LIBS=ON  -> 动态库 libyolo_detector.so
#   - BUILD_SHARED_LIBS=OFF -> 静态库 libyolo_detector.a
#
# 用法（作为库使用）：
#   add_subdirectory(cpp)
#   target_link_libraries(your_app PRIVATE yolo_detector)
#   # 需要 OpenCV / ONNX Runtime 头文件路径
# ============================================================

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

add_library(yolo_detector
    src/YOLODetector.cpp
)

target_include_directories(yolo_detector PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(yolo_detector PRIVATE
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${ONNXRUNTIME_EXTRA_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_directories(yolo_detector PRIVATE
    ${ONNXRUNTIME_LIB_DIR}
)

target_link_libraries(yolo_detector PRIVATE
    ${OpenCV_LIBS}
    onnxruntime
)

# 动态库时设置 SONAME（发布友好）
if(BUILD_SHARED_LIBS)
    set_target_properties(yolo_detector PROPERTIES
        VERSION  1.0.0
        SOVERSION 1
    )
endif()

# ============================================================
# YOLOSegmentor library
#   - BUILD_SHARED_LIBS=ON  -> libyolo_segmentor.so
#   - BUILD_SHARED_LIBS=OFF -> libyolo_segmentor.a
#
# Usage:
#   target_link_libraries(your_app PRIVATE yolo_segmentor)
# ============================================================

add_library(yolo_segmentor
    src/YOLOSegmentor.cpp
)

target_include_directories(yolo_segmentor PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(yolo_segmentor PRIVATE
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${ONNXRUNTIME_EXTRA_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_directories(yolo_segmentor PRIVATE
    ${ONNXRUNTIME_LIB_DIR}
)

target_link_libraries(yolo_segmentor PRIVATE
    ${OpenCV_LIBS}
    onnxruntime
)

if(BUILD_SHARED_LIBS)
    set_target_properties(yolo_segmentor PROPERTIES
        VERSION  1.0.0
        SOVERSION 1
    )
endif()


# ============================================================
# 可执行程序（示例 / 基准工具）
# ============================================================

add_executable(yolo_cpp
    main.cpp
)

target_include_directories(yolo_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_directories(yolo_cpp PRIVATE
    ${ONNXRUNTIME_LIB_DIR}
)

target_link_libraries(yolo_cpp PRIVATE
    yolo_detector
    ${OpenCV_LIBS}
    onnxruntime
)

# ============================================================
# Runtime RPATH
#
# 让程序运行时自动寻找：
#
# /opt/onnxruntime/lib
#
# 不需要每次手动设置 LD_LIBRARY_PATH
# ============================================================

set_target_properties(yolo_cpp PROPERTIES

    BUILD_RPATH
        "${ONNXRUNTIME_LIB_DIR}"

    INSTALL_RPATH
        "${ONNXRUNTIME_LIB_DIR}"
)


# ============================================================
# Executable: segmentation model example / benchmark tool
# ============================================================

add_executable(yolo_seg
    seg_main.cpp
)

target_include_directories(yolo_seg PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_directories(yolo_seg PRIVATE
    ${ONNXRUNTIME_LIB_DIR}
)

target_link_libraries(yolo_seg PRIVATE
    yolo_segmentor
    ${OpenCV_LIBS}
    onnxruntime
)

set_target_properties(yolo_seg PROPERTIES

    BUILD_RPATH
        "${ONNXRUNTIME_LIB_DIR}"

    INSTALL_RPATH
        "${ONNXRUNTIME_LIB_DIR}"
)
