"""
数据包捕获模块（修复版）
"""
import time
import queue
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict, Any

from ..utils.logger import LoggerMixin
from ..utils.helpers import get_flow_id
from ..utils.platform_info import get_default_interface, is_windows

try:
    from scapy.all import AsyncSniffer, IP, TCP, UDP, ICMP, Raw
    HAS_SCAPY = True
except ImportError:
    HAS_SCAPY = False


@dataclass
class PacketInfo:
    """数据包信息"""
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    length: int
    flags: str = ""
    payload_bytes: bytes = field(default_factory=bytes)
    flow_id: str = ""
    direction: str = "fwd"
    ttl: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class PacketCapture(LoggerMixin):
    """跨平台数据包捕获器"""

    def __init__(
        self,
        interface: str = "auto",
        bpf_filter: str = "ip",
        buffer_size: int = 65536,
        promiscuous: bool = True,
        queue_size: int = 10000,
    ):
        super().__init__()

        if not HAS_SCAPY:
            raise ImportError("Scapy 未安装，请执行: pip install scapy")

        self.bpf_filter = bpf_filter
        self.promiscuous = promiscuous
        self.interface = self._resolve_interface(interface)

        self._is_capturing = False
        self._sniffer: Optional['AsyncSniffer'] = None
        self._packet_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._callbacks: List[Callable] = []

        self._stats = {
            'packets_captured': 0,
            'packets_dropped': 0,
            'packets_processed': 0,
            'start_time': None,
        }

    def _resolve_interface(self, interface: str) -> str:
        if interface in ["auto", "Auto", "", None]:
            default = get_default_interface()
            return default if default else ("WLAN" if is_windows() else "eth0")
        return interface.strip()

    def register_callback(self, callback: Callable) -> None:
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def unregister_callback(self, callback: Callable) -> bool:
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            return True
        return False

    def _extract_packet_info(self, packet) -> Optional[PacketInfo]:
        try:
            if IP not in packet:
                return None
            ip = packet[IP]

            if TCP in packet:
                t, proto, flags, sport, dport = packet[TCP], "TCP", str(packet[TCP].flags), packet[TCP].sport, packet[TCP].dport
            elif UDP in packet:
                t, proto, flags, sport, dport = packet[UDP], "UDP", "", packet[UDP].sport, packet[UDP].dport
            elif ICMP in packet:
                proto, flags, sport, dport = "ICMP", "", 0, 0
            else:
                proto, flags, sport, dport = "OTHER", "", 0, 0

            payload = bytes(packet[Raw].load) if Raw in packet else b""
            flow_id, direction = get_flow_id(ip.src, ip.dst, sport, dport, proto)

            return PacketInfo(
                timestamp=float(packet.time),
                src_ip=ip.src, dst_ip=ip.dst,
                src_port=sport, dst_port=dport,
                protocol=proto, length=len(packet),
                flags=flags, payload_bytes=payload,
                flow_id=flow_id, direction=direction,
                ttl=getattr(ip, 'ttl', 0),
            )
        except Exception as e:
            self.logger.debug(f"提取包信息失败: {e}")
            return None

    def _packet_handler(self, packet):
        pkt_info = self._extract_packet_info(packet)
        if not pkt_info:
            return

        self._stats['packets_captured'] += 1

        try:
            self._packet_queue.put_nowait(pkt_info)
        except queue.Full:
            self._stats['packets_dropped'] += 1
            return

        for cb in self._callbacks:
            try:
                cb(pkt_info)
            except Exception as e:
                self.logger.error(f"回调错误: {e}")

        self._stats['packets_processed'] += 1

    def start_live_capture(self, packet_count: int = 0, timeout: Optional[float] = None):
        """使用 AsyncSniffer 启动抓包（可随时停止）"""
        if self._is_capturing:
            return
        self._is_capturing = True
        self._stats['start_time'] = time.time()

        self._sniffer = AsyncSniffer(
            iface=self.interface, filter=self.bpf_filter,
            prn=self._packet_handler, count=packet_count,
            timeout=timeout, store=False, promisc=self.promiscuous,
        )
        self._sniffer.start()
        self.logger.info(f"抓包已启动 → {self.interface}")

    def stop_capture(self):
        """停止抓包"""
        if not self._is_capturing:
            return
        self._is_capturing = False
        if self._sniffer:
            self._sniffer.stop()
            self._sniffer = None
            self.logger.info("抓包已停止")

    def get_queue_size(self) -> int:
        return self._packet_queue.qsize()

    def get_stats(self) -> Dict[str, Any]:
        return dict(self._stats)

    def get_packet_from_queue(self, timeout: Optional[float] = None) -> Optional[PacketInfo]:
        try:
            return self._packet_queue.get(timeout=timeout)
        except queue.Empty:
            return None


def create_packet_capture(config=None) -> PacketCapture:
    if config and hasattr(config, 'capture'):
        return PacketCapture(
            interface=config.capture.interface,
            bpf_filter=config.capture.bpf_filter,
            buffer_size=config.capture.buffer_size,
            promiscuous=config.capture.promiscuous,
            queue_size=config.capture.queue_size,
        )
    return PacketCapture()
