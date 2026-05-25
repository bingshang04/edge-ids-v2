"""
平台检测与系统信息
"""
import os
import sys
import platform
import psutil
from typing import Optional, List, Dict, Any, Tuple


def is_windows() -> bool:
    return sys.platform.startswith('win')


def is_linux() -> bool:
    return sys.platform.startswith('linux')


def get_platform_type() -> str:
    if is_windows():
        return 'windows'
    elif is_linux():
        machine = platform.machine().lower()
        if 'arm' in machine or 'aarch64' in machine:
            return 'raspberry_pi'
        return 'linux'
    return 'unknown'


def get_default_interface() -> Optional[str]:
    """获取默认活跃网络接口"""
    try:
        stats = psutil.net_if_stats()
        addrs = psutil.net_if_addrs()

        # 优先级：以太网 > WLAN
        if is_windows():
            priority = ['ethernet', 'wi-fi', 'wlan', '以太网']
        else:
            priority = ['eth0', 'ens33', 'enp0s3', 'wlan0', 'wlp2s0']

        # 按优先级查找
        for prefix in priority:
            for name, stat in stats.items():
                if prefix.lower() in name.lower() and stat.isup:
                    return name

        # 返回第一个可用接口
        for name, stat in stats.items():
            if stat.isup and not name.startswith('lo'):
                return name
    except Exception:
        pass
    return None if not is_windows() else "WLAN"


def list_network_interfaces() -> None:
    """列出所有网络接口"""
    try:
        stats = psutil.net_if_stats()
        addrs = psutil.net_if_addrs()
        print("=" * 60)
        print("可用网络接口:")
        print("=" * 60)
        for name, stat in stats.items():
            status = "已连接" if stat.isup else "未连接"
            ips = [a.address for a in addrs.get(name, []) if a.family == 2]
            print(f"  {name}: {status}, IP: {', '.join(ips) if ips else '无'}")
        default = get_default_interface()
        if default:
            print(f"\n推荐接口: {default}")
        print("=" * 60)
    except Exception as e:
        print(f"获取网络接口失败: {e}")


def check_admin() -> bool:
    """检查是否具有管理员/root权限"""
    try:
        if is_windows():
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin()
        else:
            return os.geteuid() == 0
    except Exception:
        return False


def get_system_info() -> Dict[str, Any]:
    """获取系统信息"""
    mem = psutil.virtual_memory()
    return {
        'platform': get_platform_type(),
        'machine': platform.machine(),
        'processor': platform.processor() or "Unknown",
        'system': f"{platform.system()} {platform.release()}",
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_gb': round(mem.total / (1024 ** 3), 2),
    }


def get_resource_usage() -> Dict[str, Any]:
    """获取系统资源使用情况"""
    mem = psutil.virtual_memory()
    return {
        'cpu_percent': psutil.cpu_percent(interval=0.1),
        'memory_used_gb': round(mem.used / (1024 ** 3), 2),
        'memory_percent': mem.percent,
    }
