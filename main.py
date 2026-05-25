

"""
Edge-IDS v2.0 主程序入口
基于 ECA-TCN 的轻量级边缘入侵检测系统
"""
import sys
import time
import argparse
from collections import deque
from pathlib import Path
from typing import Optional, List

ROOT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

from src.config import get_settings
from src.utils.logger import setup_logging, get_logger
from src.utils.platform_info import (
    is_windows, check_admin, get_default_interface, list_network_interfaces,
    get_platform_type,
)
from src.capture.packet_capture import PacketCapture, create_packet_capture
from src.features.flow_features import FeatureExtractor, create_feature_extractor
from src.inference.detector import IDSDetector, create_detector

# Web 模块（可选）
try:
    from src.web.dashboard import create_dashboard
    HAS_DASHBOARD = True
except ImportError:
    HAS_DASHBOARD = False
    create_dashboard = None


class EdgeIDS:
    """Edge-IDS v2.0 主控制器"""

    def __init__(self, config_path: Optional[str] = None):
        self.config = get_settings(config_path)
        setup_logging(level=self.config.log.level, log_dir=self.config.log.log_dir)
        self.logger = get_logger(__name__)

        self.capture: Optional[PacketCapture] = None
        self.extractor: Optional[FeatureExtractor] = None
        self.detector: Optional[IDSDetector] = None
        self.dashboard: Optional[object] = None

        self._is_running = False
        self._start_time: Optional[float] = None

        self._stats = {
            'packets_processed': 0,
            'flows_analyzed': 0,
            'attacks_detected': 0,
            'total_attacks': 0,
        }

        # 攻击记录（最近 200 条）
        self._attack_log: deque = deque(maxlen=200)

        self.logger.info(f"Edge-IDS v2.0 初始化 | 平台: {get_platform_type()}")
        self.logger.info(f"模型: ECA-TCN | 特征维度: {self.config.feature.feature_dim}")

    def initialize(self) -> 'EdgeIDS':
        """初始化所有组件"""
        self.logger.info("正在初始化组件...")

        # 网络接口
        if self.config.capture.interface in ["auto", "Auto", None, ""]:
            default_iface = get_default_interface()
            if default_iface:
                self.config.capture.interface = default_iface
                self.logger.info(f"自动检测接口 → {default_iface}")
            else:
                self.config.capture.interface = "WLAN" if is_windows() else "eth0"

        # 初始化各组件
        self.capture = create_packet_capture(self.config)
        self.logger.info(f"捕获器就绪 → {self.config.capture.interface}")

        self.extractor = create_feature_extractor(self.config)
        self.logger.info(f"特征提取器就绪 → {self.config.feature.feature_dim}维")

        self.detector = create_detector(self.config)
        self.logger.info(f"检测器就绪 → {'ECA-TCN' if self.config.model.use_eca else 'TCN'}")

        return self

    def _packet_callback(self, packet_info):
        """数据包回调：提取特征 → 推理检测"""
        try:
            features = self.extractor.process_packet(packet_info)
            if features is not None:
                self._stats['flows_analyzed'] += 1
                result = self.detector.predict(features)

                if result.is_attack(self.config.inference.alert_threshold):
                    # 模型直接输出攻击类型（多分类）
                    attack_type = self._classify_attack(result)

                    # 规则过滤：对低风险类型 + 已建立连接的情况进行降敏
                    if self._filter_false_positive(packet_info, attack_type):
                        self._update_dashboard()
                        self._stats['packets_processed'] += 1
                        return

                    self._stats['attacks_detected'] += 1
                    self._stats['total_attacks'] += 1

                    # 危险等级（基于攻击类型映射表）
                    danger = self._danger_level(result)

                    record = {
                        'time': time.strftime('%H:%M:%S'),
                        'src': f"{packet_info.src_ip}:{packet_info.src_port}",
                        'dst': f"{packet_info.dst_ip}:{packet_info.dst_port}",
                        'protocol': packet_info.protocol,
                        'type': attack_type,
                        'attack_type_id': result.attack_type_id,
                        'confidence': round(result.confidence, 4),
                        'danger': danger,
                        'latency_ms': round(result.latency_ms, 2),
                    }
                    self._attack_log.appendleft(record)  # 最新的在前面

                    # 同步攻击记录到 dashboard
                    if self.dashboard:
                        self.dashboard.add_attack(record)

                    self.logger.warning(
                        f"⚠ 检测到攻击[{attack_type}] | 危险等级: {danger} | "
                        f"来源: {record['src']} → {record['dst']} | "
                        f"置信度: {result.confidence:.4f}"
                    )

                self._update_dashboard()
            self._stats['packets_processed'] += 1
        except Exception as e:
            self.logger.error(f"数据包处理错误: {e}")

    def _classify_attack(self, result) -> str:
        """直接返回模型输出的攻击类型名称（多分类模型）"""
        if hasattr(result, 'attack_type_name') and result.attack_type_name:
            return result.attack_type_name
        # 降级：通过检测器的映射表查询
        if hasattr(result, 'attack_type_id') and hasattr(self.detector, '_attack_type_names'):
            return self.detector._attack_type_names.get(
                result.attack_type_id, f'Unknown-{result.attack_type_id}')
        return 'Unknown'

    def _filter_false_positive(self, pkt, attack_type: str) -> bool:
        """规则过滤：模型误判正常流量为攻击时的降敏处理"""
        if attack_type in ('Reconnaissance', 'Analysis'):
            ack = getattr(pkt, 'flow_ack_flags', 0)
            dpkts = getattr(pkt, 'flow_dpkts', 0)
            if ack > 0 or dpkts > 0:
                return True
        return False

    def _danger_level(self, result) -> str:
        """根据攻击类型返回危险等级（使用映射表，非置信度推测）"""
        if hasattr(result, 'danger_level') and result.danger_level:
            return result.danger_level
        # 降级：通过检测器的映射表查询
        if hasattr(result, 'attack_type_id') and hasattr(self.detector, '_attack_danger_levels'):
            return self.detector._attack_danger_levels.get(
                result.attack_type_id, '未知')
        return '未知'

    def _update_dashboard(self):
        """更新仪表盘状态"""
        if not self.dashboard:
            return
        try:
            detector_stats = self.detector.get_stats()
            self.dashboard.update_status(
                is_running=self._is_running,
                packets_captured=self.capture._stats.get('packets_captured', 0),
                packets_dropped=self.capture._stats.get('packets_dropped', 0),
                flows_analyzed=self._stats['flows_analyzed'],
                flows_active=self.extractor.active_flow_count,
                attacks_detected=self._stats['attacks_detected'],
                attacks_total=self._stats['total_attacks'],
                avg_latency_ms=detector_stats.get('avg_latency_ms', 0),
                start_time=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self._start_time))
                if self._start_time else None,
            )
        except Exception:
            pass

    def start_detection(self):
        """开始实时检测"""
        if self._is_running:
            return
        self.logger.info(f"开始实时检测 → {self.config.capture.interface}")
        self.capture.register_callback(self._packet_callback)
        self.capture.start_live_capture()
        self._is_running = True
        self._start_time = time.time()

    def stop_detection(self):
        """停止检测"""
        if not self._is_running:
            return
        self.logger.info("正在停止检测...")
        if self.capture:
            self.capture.stop_capture()
        self._is_running = False
        # 强制更新仪表盘状态为已停止
        self._update_dashboard()
        self.logger.info("检测已停止")

    def start_dashboard(self):
        """启动 Web 仪表盘"""
        if not HAS_DASHBOARD:
            self.logger.warning("Web 仪表盘不可用（缺少 Flask）")
            return
        self.dashboard = create_dashboard(self.config)
        self.dashboard.register_callback('start', self.start_detection)
        self.dashboard.register_callback('stop', self.stop_detection)
        self.dashboard.run(threaded=True)
        self.logger.info(f"仪表盘已启动: http://{self.config.web.host}:{self.config.web.port}")

    def run(self, mode: str = 'full', interface: Optional[str] = None):
        """运行系统"""
        if interface:
            self.config.capture.interface = interface
            if self.capture:
                self.capture.stop_capture()
            self.capture = create_packet_capture(self.config)

        try:
            if mode == 'full':
                self.start_dashboard()
                self.start_detection()
            elif mode == 'dashboard':
                self.start_dashboard()
            elif mode == 'capture':
                self.start_detection()

            self.logger.info("系统运行中，按 Ctrl+C 停止...")
            while True:
                if self._is_running:
                    self._update_dashboard()
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("收到中断信号")
        finally:
            self.shutdown()

    def shutdown(self):
        """关闭系统"""
        self.stop_detection()
        duration = time.time() - self._start_time if self._start_time else 0
        self.logger.info("=" * 50)
        self.logger.info(f"Edge-IDS v2.0 运行统计")
        self.logger.info(f"  运行时长: {duration:.1f}s")
        self.logger.info(f"  处理数据包: {self._stats['packets_processed']}")
        self.logger.info(f"  分析流: {self._stats['flows_analyzed']}")
        self.logger.info(f"  检测攻击: {self._stats['total_attacks']}")
        if self.detector:
            stats = self.detector.get_stats()
            self.logger.info(f"  平均推理延迟: {stats.get('avg_latency_ms', 0):.2f}ms")
        self.logger.info("=" * 50)
        self.logger.info("系统已关闭")


def main():
    parser = argparse.ArgumentParser(description='Edge-IDS v2.0 边缘入侵检测系统（ECA-TCN）')
    parser.add_argument('--mode', choices=['full', 'capture', 'dashboard'], default='full',
                        help='运行模式（默认: full）')
    parser.add_argument('--interface', default=None, help='指定网络接口')
    parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    parser.add_argument('--list-interfaces', action='store_true', help='列出网络接口')
    args = parser.parse_args()

    if args.list_interfaces:
        list_network_interfaces()
        return

    # 捕获模式需要管理员权限
    if args.mode != 'dashboard':
        if not check_admin():
            print("⚠ 数据包捕获需要管理员/root权限！")
            if is_windows():
                print("  请以管理员身份运行 PowerShell/CMD")
            else:
                print("  请使用 sudo 运行")
            sys.exit(1)

    try:
        ids = EdgeIDS(config_path=args.config)
        ids.initialize()
        ids.run(mode=args.mode, interface=args.interface)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
