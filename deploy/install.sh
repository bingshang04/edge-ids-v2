#!/bin/bash
# ===========================================================================
# Edge-IDS v2.0 一键安装脚本（树莓派5 / Debian / Ubuntu）
# ===========================================================================
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

PROJECT_NAME="Edge-IDS v2.0"
INSTALL_DIR="/opt/edge-ids"
VENV_DIR="${INSTALL_DIR}/venv"
SERVICE_NAME="edge-ids"
LOG_DIR="${INSTALL_DIR}/logs"
DATA_DIR="${INSTALL_DIR}/data"
MODEL_DIR="${DATA_DIR}/models"

# 默认 Python 版本
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo -e "${CYAN}================================================${NC}"
echo -e "${CYAN}   ${PROJECT_NAME} 安装脚本${NC}"
echo -e "${CYAN}================================================${NC}"
echo ""

# ------------------------------------------------------------------
# 0. 权限检查
# ------------------------------------------------------------------
if [[ $EUID -ne 0 ]]; then
    echo -e "${RED}[错误] 请使用 root 权限运行此脚本: sudo bash install.sh${NC}"
    exit 1
fi

# ------------------------------------------------------------------
# 1. 系统依赖检查
# ------------------------------------------------------------------
echo -e "${YELLOW}[1/7] 检查系统依赖...${NC}"

ARCH=$(uname -m)
echo "  - 架构: ${ARCH}"

# 树莓派检测
if [[ "$ARCH" == "aarch64" ]] || [[ "$ARCH" == "armv7l" ]]; then
    IS_RPI=true
    echo -e "  ${GREEN}- 检测到 ARM 架构（树莓派）${NC}"
else
    IS_RPI=false
    echo -e "  ${YELLOW}- 非 ARM 架构，部分优化将跳过${NC}"
fi

# 内存检查
TOTAL_MEM=$(free -m | awk '/^Mem:/{print $2}')
echo "  - 总内存: ${TOTAL_MEM} MB"
if [[ $TOTAL_MEM -lt 3500 ]]; then
    echo -e "  ${YELLOW}! 内存小于 3.5GB，系统可能不稳定${NC}"
fi

# 安装系统依赖
echo "  - 安装系统包..."
apt-get update -qq
apt-get install -y -qq \
    iptables \
    tcpdump \
    libcap-dev \
    python3-dev \
    python3-pip \
    python3-venv \
    git \
    curl \
    htop \
    > /dev/null 2>&1

echo -e "  ${GREEN}系统依赖安装完成${NC}"

# ------------------------------------------------------------------
# 2. 创建目录结构
# ------------------------------------------------------------------
echo -e "${YELLOW}[2/7] 创建目录结构...${NC}"

mkdir -p "${INSTALL_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "${MODEL_DIR}"
mkdir -p "${DATA_DIR}"

# 复制项目文件（如果从源码目录运行）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_SRC="$(dirname "${SCRIPT_DIR}")"

if [[ -f "${PROJECT_SRC}/main.py" ]]; then
    echo "  - 从 ${PROJECT_SRC} 复制项目文件..."
    rsync -a --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
          --exclude='.claude' --exclude='venv' --exclude='logs' \
          "${PROJECT_SRC}/" "${INSTALL_DIR}/"
else
    echo -e "  ${YELLOW}! 未检测到项目源码，假设已存在于 ${INSTALL_DIR}${NC}"
fi

echo -e "  ${GREEN}目录结构创建完成${NC}"

# ------------------------------------------------------------------
# 3. 创建虚拟环境
# ------------------------------------------------------------------
echo -e "${YELLOW}[3/7] 创建 Python 虚拟环境...${NC}"

if [[ ! -d "${VENV_DIR}" ]]; then
    ${PYTHON_BIN} -m venv "${VENV_DIR}"
    echo -e "  ${GREEN}虚拟环境已创建: ${VENV_DIR}${NC}"
else
    echo "  - 虚拟环境已存在，跳过"
fi

# 激活虚拟环境
source "${VENV_DIR}/bin/activate"

# 升级 pip
pip install --upgrade pip -q

echo -e "  ${GREEN}虚拟环境配置完成${NC}"

# ------------------------------------------------------------------
# 4. 安装 Python 依赖
# ------------------------------------------------------------------
echo -e "${YELLOW}[4/7] 安装 Python 依赖...${NC}"

# 核心依赖（不含 torch，树莓派上用 TFLite）
pip install -q \
    numpy>=1.21.0 \
    pandas>=1.3.0 \
    scikit-learn>=1.0.0 \
    scapy>=2.4.5 \
    flask>=2.0.0 \
    flask-cors>=3.0.0 \
    pyyaml>=5.4.0 \
    psutil>=5.8.0 \
    joblib>=1.0.0

# TFLite 运行时（树莓派专用）
if [[ "$IS_RPI" == true ]]; then
    echo "  - 安装 tflite-runtime（ARM64）..."
    pip install -q tflite-runtime>=2.14.0 2>/dev/null || {
        echo -e "  ${YELLOW}! tflite-runtime 安装失败，尝试从源码安装...${NC}"
        echo "  - 请手动安装: pip install tflite-runtime"
    }
else
    echo "  - PC 环境，跳过 tflite-runtime（开发时用 TensorFlow）"
fi

echo -e "  ${GREEN}依赖安装完成${NC}"

# ------------------------------------------------------------------
# 5. 配置 iptables
# ------------------------------------------------------------------
echo -e "${YELLOW}[5/7] 配置 iptables 规则...${NC}"

# 确保 EDGE_IDS 链不存在（避免重复安装冲突）
if iptables -L EDGE_IDS -n > /dev/null 2>&1; then
    echo "  - EDGE_IDS 链已存在，跳过初始化"
else
    echo "  - EDGE_IDS 链将由系统运行时自动创建"
fi

echo -e "  ${GREEN}iptables 配置完成${NC}"

# ------------------------------------------------------------------
# 6. 安装 systemd 服务
# ------------------------------------------------------------------
echo -e "${YELLOW}[6/7] 安装 systemd 服务...${NC}"

# 替换服务文件中的路径占位符
SERVICE_SRC="${INSTALL_DIR}/deploy/edge-ids.service"
SERVICE_DST="/etc/systemd/system/${SERVICE_NAME}.service"

if [[ -f "${SERVICE_SRC}" ]]; then
    cp "${SERVICE_SRC}" "${SERVICE_DST}"
else
    # 直接从脚本目录复制
    if [[ -f "${SCRIPT_DIR}/edge-ids.service" ]]; then
        cp "${SCRIPT_DIR}/edge-ids.service" "${SERVICE_DST}"
    else
        echo -e "  ${YELLOW}! 未找到 edge-ids.service 文件，跳过${NC}"
    fi
fi

if [[ -f "${SERVICE_DST}" ]]; then
    # 更新路径
    sed -i "s|WorkingDirectory=/opt/edge-ids|WorkingDirectory=${INSTALL_DIR}|g" "${SERVICE_DST}"
    sed -i "s|/opt/edge-ids|${INSTALL_DIR}|g" "${SERVICE_DST}"

    systemctl daemon-reload
    systemctl enable "${SERVICE_NAME}"
    echo -e "  ${GREEN}systemd 服务已安装并启用${NC}"
    echo ""
    echo -e "  ${CYAN}管理命令:${NC}"
    echo "    sudo systemctl start   ${SERVICE_NAME}   # 启动"
    echo "    sudo systemctl stop    ${SERVICE_NAME}   # 停止"
    echo "    sudo systemctl status  ${SERVICE_NAME}   # 状态"
    echo "    sudo journalctl -u     ${SERVICE_NAME} -f  # 日志"
fi

# ------------------------------------------------------------------
# 7. 验证安装
# ------------------------------------------------------------------
echo -e "${YELLOW}[7/7] 验证安装...${NC}"

echo "  - Python: $(${VENV_DIR}/bin/python --version)"
echo "  - pip: $(${VENV_DIR}/bin/pip --version | head -1)"
echo "  - 安装目录: ${INSTALL_DIR}"

# 检查关键文件
CHECK_FILES=(
    "${INSTALL_DIR}/main.py"
    "${INSTALL_DIR}/config.yaml"
    "${INSTALL_DIR}/src/inference/factory.py"
)

ALL_OK=true
for f in "${CHECK_FILES[@]}"; do
    if [[ -f "$f" ]]; then
        echo -e "  ${GREEN}✓${NC} $f"
    else
        echo -e "  ${RED}✗${NC} $f (缺失)"
        ALL_OK=false
    fi
done

# 检查模型文件
if [[ -f "${MODEL_DIR}/tcn_model.tflite" ]]; then
    echo -e "  ${GREEN}✓${NC} TFLite 模型: ${MODEL_DIR}/tcn_model.tflite"
else
    echo -e "  ${YELLOW}!${NC} TFLite 模型文件不存在: ${MODEL_DIR}/tcn_model.tflite"
    echo "    请先将模型文件放置到 ${MODEL_DIR}/ 目录"
fi

# ------------------------------------------------------------------
# 完成
# ------------------------------------------------------------------
echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}   ${PROJECT_NAME} 安装完成！${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo -e "  安装目录: ${INSTALL_DIR}"
echo -e "  日志目录: ${LOG_DIR}"
echo -e "  模型目录: ${MODEL_DIR}"
echo ""

if [[ "$ALL_OK" == true ]]; then
    echo -e "  启动服务: ${CYAN}sudo systemctl start ${SERVICE_NAME}${NC}"
    echo -e "  查看日志: ${CYAN}sudo journalctl -u ${SERVICE_NAME} -f${NC}"
    echo -e "  Web 仪表盘: ${CYAN}http://<树莓派IP>:8080${NC}"
else
    echo -e "  ${YELLOW}部分文件缺失，请检查安装${NC}"
fi
echo ""
