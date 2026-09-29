#!/bin/bash
# ============================================================
# YOLO dev 容器一键启动脚本（WSL / Linux bash 版）
#
# 用法:
#   ./docker/run_wsl.sh                # 启动/进入容器
#   ./docker/run_wsl.sh --recreate     # 强制删除并重建容器
#
# 说明:
#   - 容器以后台方式常驻（sleep infinity），再通过 docker exec 进入，
#     避免 docker run -it 附加到不读输入的进程导致的"卡住无响应"
#   - 已启用 GPU 透传 (--gpus all) 与共享内存 8G (--shm-size=8g)
# ============================================================

set -e

# 容器名与镜像名
NAME="${CONTAINER_NAME:-yoloworkbench}"
IMAGE="${IMAGE:-yoloworkbench:cuda12.8}"

# 项目根目录（脚本位于 docker/ 下，取上级目录）
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ------------------------------------------------------------
# 0. 检查 docker 可用性
# ------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] docker command not found. Please start Docker Desktop (WSL2 backend)." >&2
    exit 1
fi

# ------------------------------------------------------------
# 1. 检查镜像是否存在
# ------------------------------------------------------------
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[ERROR] Image not found: $IMAGE" >&2
    echo "        Build it first:" >&2
    echo "        docker build -t $IMAGE -f docker/Dockerfile ." >&2
    exit 1
fi

# ------------------------------------------------------------
# 2. 处理 --recreate 参数
# ------------------------------------------------------------
if [ "$1" = "--recreate" ]; then
    echo "[WARN] Removing existing container: $NAME"
    docker rm -f "$NAME" 2>/dev/null || true
fi

# ------------------------------------------------------------
# 3. 容器不存在则创建
# ------------------------------------------------------------
if ! docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "[OK] Creating container: $NAME"
    docker run -d \
        --name "$NAME" \
        --gpus all \
        --shm-size=8g \
        -v "$PROJECT_ROOT:/workspace" \
        -w /workspace \
        "$IMAGE" \
        sleep infinity
    echo "[OK] Mount: $PROJECT_ROOT -> /workspace"
elif [ "$(docker inspect -f '{{.State.Running}}' "$NAME")" != "true" ]; then
    echo "[OK] Container exists but stopped, starting: $NAME"
    docker start "$NAME"
else
    echo "[OK] Container already running: $NAME"
fi

# ------------------------------------------------------------
# 4. 进入容器
# ------------------------------------------------------------
echo "[OK] Entering container: $NAME"
echo "----------------------------------------"
docker exec -it "$NAME" bash
