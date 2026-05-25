"""辅助工具函数"""
import os
from pathlib import Path
from typing import Tuple


def ensure_dir(path: str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator != 0 else default


def clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


def format_bytes(size_bytes: float) -> str:
    if size_bytes == 0:
        return "0 B"
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(size_bytes)
    idx = 0
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024
        idx += 1
    return f"{size:.2f} {units[idx]}"


def get_flow_id(
    src_ip: str, dst_ip: str,
    src_port: int, dst_port: int,
    protocol: str
) -> Tuple[str, str]:
    """生成双向流ID和方向"""
    if (src_ip, src_port) < (dst_ip, dst_port):
        flow_id = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}"
        direction = "fwd"
    else:
        flow_id = f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{protocol}"
        direction = "bwd"
    return flow_id, direction
