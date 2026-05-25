#!/bin/bash
# Edge-IDS v2.0 启动脚本 (Linux/Mac)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="full"
INTERFACE=""
CONFIG="config.yaml"

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode) MODE="$2"; shift 2 ;;
        --interface) INTERFACE="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --help)
            echo "Edge-IDS v2.0 (ECA-TCN)"
            echo "用法: ./run.sh [--mode full|capture|dashboard] [--interface eth0] [--config config.yaml]"
            exit 0 ;;
        *) echo "未知选项: $1"; exit 1 ;;
    esac
done

if [ -d "venv" ]; then
    source venv/bin/activate
fi

if [ "$MODE" != "dashboard" ] && [ "$EUID" -ne 0 ]; then
    echo "需要 root 权限，正在提权..."
    exec sudo "$0" "$@"
fi

echo "Edge-IDS v2.0 启动 | 模式: $MODE | 配置: $CONFIG"
python main.py --mode "$MODE" --config "$CONFIG" ${INTERFACE:+--interface "$INTERFACE"}
