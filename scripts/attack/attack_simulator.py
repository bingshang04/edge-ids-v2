"""
Edge-IDS v2.0 攻击模拟工具
用于生成攻击流量以测试入侵检测系统的实时检测能力。
"""
import os
import sys
import time
import random
import signal
import socket
import threading
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

# 路径注入（与 train.py 保持一致）
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, ROOT_DIR)

from src.utils.logger import LoggerMixin, get_logger
from src.utils.platform_info import (
    check_admin, get_default_interface, list_network_interfaces,
    is_windows,
)

try:
    from scapy.all import IP, TCP, UDP, Raw, sendp, Ether
    HAS_SCAPY = True
except ImportError:
    HAS_SCAPY = False

logger = get_logger(__name__)


# ==================== 统计计数器 ====================

@dataclass
class AttackStats:
    """线程安全的攻击统计"""
    packets_sent: int = 0
    bytes_sent: int = 0
    errors: int = 0
    start_time: float = 0.0

    _lock: threading.Lock = None

    def __post_init__(self):
        self._lock = threading.Lock()

    def inc_packets(self, n: int = 1, nbytes: int = 0):
        with self._lock:
            self.packets_sent += n
            self.bytes_sent += nbytes

    def inc_errors(self, n: int = 1):
        with self._lock:
            self.errors += n

    def snapshot(self) -> dict:
        with self._lock:
            return {
                'packets_sent': self.packets_sent,
                'bytes_sent': self.bytes_sent,
                'errors': self.errors,
                'elapsed': time.time() - self.start_time if self.start_time else 0,
            }


# ==================== 攻击生成器 ====================

def _get_local_ip() -> str:
    """获取本机首选 LAN IP"""
    import psutil
    for name, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET and not addr.address.startswith('127.'):
                # 优先返回有默认网关的接口 IP
                return addr.address
    return socket.gethostbyname(socket.gethostname())


def _format_pkt(n: int) -> str:
    """格式化数据包计数（带 K/M 单位）"""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def _run_syn_flood(stop_event: threading.Event, stats: AttackStats, cfg: dict):
    """SYN Flood 攻击 — 模拟 DoS"""
    target = cfg['target']
    pps = cfg['pps']
    finish = cfg.get('finish', False)
    interval = 1.0 / pps

    while not stop_event.is_set():
        src_port = random.randint(1024, 65535)
        dst_port = random.randint(1, 65535)

        try:
            pkt = Ether() / IP(dst=target) / TCP(sport=src_port, dport=dst_port, flags='S')
            sendp(pkt, iface=cfg['iface'], verbose=0)
            stats.inc_packets(1, len(pkt))

            if finish:
                # 立即发送 RST 完成流，触发特征提取
                rst = Ether() / IP(dst=target) / TCP(sport=src_port, dport=dst_port, flags='R')
                sendp(rst, iface=cfg['iface'], verbose=0)
                stats.inc_packets(1, len(rst))
        except Exception as e:
            stats.inc_errors()

        time.sleep(interval)


def _run_port_scan(stop_event: threading.Event, stats: AttackStats, cfg: dict):
    """端口扫描 — 模拟 Reconnaissance"""
    target = cfg['target']
    port_start, port_end = cfg['port_range']
    pps = cfg['pps']
    finish = cfg.get('finish', False)
    interval = 1.0 / pps

    current = port_start
    while not stop_event.is_set():
        src_port = random.randint(1024, 65535)

        try:
            pkt = Ether() / IP(dst=target) / TCP(sport=src_port, dport=current, flags='S')
            sendp(pkt, iface=cfg['iface'], verbose=0)
            stats.inc_packets(1, len(pkt))

            if finish:
                rst = Ether() / IP(dst=target) / TCP(sport=src_port, dport=current, flags='R')
                sendp(rst, iface=cfg['iface'], verbose=0)
                stats.inc_packets(1, len(rst))
        except Exception as e:
            stats.inc_errors()

        # 循环扫描端口范围
        current = port_start + ((current - port_start + 1) % (port_end - port_start + 1))
        time.sleep(interval)


def _run_udp_flood(stop_event: threading.Event, stats: AttackStats, cfg: dict):
    """UDP Flood — 高速 UDP 数据包"""
    target = cfg['target']
    dst_port = cfg['port']
    pps = cfg['pps']
    payload_size = cfg.get('payload_size', 64)
    interval = 1.0 / pps

    payload = b'\x00' * payload_size

    while not stop_event.is_set():
        src_port = random.randint(1024, 65535)

        try:
            pkt = Ether() / IP(dst=target) / UDP(sport=src_port, dport=dst_port) / Raw(load=payload)
            sendp(pkt, iface=cfg['iface'], verbose=0)
            stats.inc_packets(1, len(pkt))
        except Exception as e:
            stats.inc_errors()

        time.sleep(interval)


ATTACK_RUNNERS = {
    'syn_flood': _run_syn_flood,
    'port_scan': _run_port_scan,
    'udp_flood': _run_udp_flood,
}


# ==================== 主控制器 ====================

class AttackSimulator(LoggerMixin):
    """攻击模拟器"""

    # 各攻击类型的默认参数
    DEFAULT_PPS = {'syn_flood': 100, 'port_scan': 50, 'udp_flood': 150}

    def __init__(self, args: argparse.Namespace):
        super().__init__()

        if not HAS_SCAPY:
            raise ImportError('Scapy 未安装，请执行: pip install scapy')

        self._stop_event = threading.Event()
        self._stats = AttackStats()

        # 解析目标 IP 和接口
        self.target = args.target
        iface = args.interface or get_default_interface() or 'WLAN'

        # 构建配置
        attack_type = args.type
        self.cfg = {
            'target': self.target,
            'iface': iface,
            'pps': args.pps or self.DEFAULT_PPS.get(attack_type, 100),
            'duration': args.duration,
            'finish': args.finish,
            'payload_size': args.payload,
        }

        if attack_type in ('syn_flood', 'udp_flood'):
            self.cfg['port'] = args.port or random.randint(1, 65535)
        if attack_type == 'port_scan':
            self.cfg['port_range'] = args.ports or (1, 1024)

        self.attack_type = attack_type
        self.attack_fn = ATTACK_RUNNERS[attack_type]

        self.logger.info(f'攻击模拟器就绪 | 类型: {attack_type} | 目标: {self.target} | 接口: {iface}')
        self.logger.info(f'速率: {self.cfg["pps"]} pps | 时长: {self.cfg["duration"]}s | '
                         f'完成流: {"是" if self.cfg["finish"] else "否"} | L2发送')

    def run(self):
        """启动攻击"""
        self._stats.start_time = time.time()
        self._stop_event.clear()

        # 攻击线程
        attack_thread = threading.Thread(
            target=self._attack_wrapper,
            daemon=True,
        )
        attack_thread.start()

        # 注册信号处理
        def _sig_handler(sig, frame):
            self.logger.info('\n收到中断信号，正在停止...')
            self._stop_event.set()

        signal.signal(signal.SIGINT, _sig_handler)

        # 统计打印循环
        try:
            last_count = 0
            last_time = time.time()
            while not self._stop_event.is_set():
                time.sleep(1.0)

                snap = self._stats.snapshot()
                current = snap['packets_sent']
                elapsed = time.time() - last_time
                rate = (current - last_count) / elapsed if elapsed > 0 else 0
                last_count = current
                last_time = time.time()

                print(f'\r[攻击: {self.attack_type}] '
                      f'已发送: {_format_pkt(current)} 包 | '
                      f'速率: {rate:.0f} pps | '
                      f'流量: {snap["bytes_sent"]/1024/1024:.2f} MB | '
                      f'已运行: {snap["elapsed"]:.0f}s',
                      end='', flush=True)

                # 达到设定时间自动停止
                if self.cfg['duration'] > 0 and snap['elapsed'] >= self.cfg['duration']:
                    self.logger.info('\n\n达到设定时间，正在停止...')
                    self._stop_event.set()

        except KeyboardInterrupt:
            self._stop_event.set()
        finally:
            print()  # 换行
            self.shutdown()

    def _attack_wrapper(self):
        """攻击线程包装 — 捕获顶层异常"""
        try:
            self.attack_fn(self._stop_event, self._stats, self.cfg)
        except Exception as e:
            self.logger.error(f'攻击线程异常: {e}')

    def shutdown(self):
        """关闭并打印摘要"""
        self._stop_event.set()
        snap = self._stats.snapshot()

        self.logger.info('=' * 50)
        self.logger.info('攻击模拟摘要')
        self.logger.info(f'  攻击类型: {self.attack_type}')
        self.logger.info(f'  目标地址: {self.target}')
        self.logger.info(f'  发送数据包: {_format_pkt(snap["packets_sent"])}')
        self.logger.info(f'  发送流量: {snap["bytes_sent"]/1024/1024:.2f} MB')
        self.logger.info(f'  运行时长: {snap["elapsed"]:.1f}s')
        self.logger.info(f'  平均速率: {snap["packets_sent"]/snap["elapsed"]:.0f} pps'
                         if snap['elapsed'] > 0 else '  平均速率: N/A')
        self.logger.info(f'  错误数: {snap["errors"]}')
        self.logger.info('=' * 50)


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(
        description='Edge-IDS v2.0 攻击模拟工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  %(prog)s --type syn_flood --finish --pps 200 --duration 30
  %(prog)s --type port_scan --ports 1 5000 --pps 100 --finish
  %(prog)s --type udp_flood --target 8.8.8.8 --port 53 --pps 150
        ''',
    )

    parser.add_argument('--type', required=True,
                        choices=['syn_flood', 'port_scan', 'udp_flood'],
                        help='攻击类型')
    parser.add_argument('--target', required=True,
                        help='目标 IP（必填，建议用路由器 IP 如 192.168.1.1 或外部 IP）')
    parser.add_argument('--port', type=int, default=None,
                        help='目标端口（syn_flood / udp_flood，默认随机）')
    parser.add_argument('--ports', type=int, nargs=2, default=None,
                        metavar=('START', 'END'),
                        help='端口扫描范围（port_scan，默认 1 1024）')
    parser.add_argument('--pps', type=int, default=None,
                        help='发包速率/秒（默认：syn_flood=100, port_scan=50, udp_flood=150）')
    parser.add_argument('--duration', type=int, default=60,
                        help='持续时间(秒)，0=手动停止（默认：60）')
    parser.add_argument('--finish', action='store_true',
                        help='发送 RST 完成流以触发即时检测（syn_flood / port_scan）')
    parser.add_argument('--interface', default=None,
                        help='发送接口（默认：自动检测，通常为 WLAN）')
    parser.add_argument('--payload', type=int, default=64,
                        help='UDP 负载大小(字节)（默认：64）')
    parser.add_argument('--list-interfaces', action='store_true',
                        help='列出可用网络接口')

    args = parser.parse_args()

    if args.list_interfaces:
        list_network_interfaces()
        return

    # 管理员权限检查
    if not check_admin():
        print('⚠ 攻击模拟需要管理员/root 权限发送原始数据包！')
        if is_windows():
            print('  请以管理员身份运行终端')
        else:
            print('  请使用 sudo 运行')
        sys.exit(1)

    try:
        sim = AttackSimulator(args)
        sim.run()
    except Exception as e:
        logger.error(f'启动失败: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
