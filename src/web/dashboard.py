"""
Web 仪表盘服务器（基于 Flask）
"""
import json
import threading
from collections import deque
from datetime import datetime
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

from ..utils.logger import LoggerMixin
from ..utils.platform_info import get_system_info, get_resource_usage


@dataclass
class SystemStatus:
    """系统状态"""
    is_running: bool = False
    packets_captured: int = 0
    packets_dropped: int = 0
    flows_analyzed: int = 0
    flows_active: int = 0
    attacks_detected: int = 0
    attacks_total: int = 0
    avg_latency_ms: float = 0.0
    start_time: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_running': self.is_running,
            'packets_captured': self.packets_captured,
            'packets_dropped': self.packets_dropped,
            'flows_analyzed': self.flows_analyzed,
            'flows_active': self.flows_active,
            'attacks_detected': self.attacks_detected,
            'attacks_total': self.attacks_total,
            'avg_latency_ms': round(self.avg_latency_ms, 2),
            'start_time': self.start_time,
        }


class DashboardServer(LoggerMixin):
    """Web 仪表盘"""

    def __init__(self, host: str = '0.0.0.0', port: int = 8080, debug: bool = False,
                 template_folder: str = 'templates', static_folder: str = 'static'):
        super().__init__()
        self.host = host
        self.port = port

        self.app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
        CORS(self.app)

        self._status = SystemStatus()
        self._status_lock = threading.Lock()
        self._callbacks: Dict[str, Callable] = {}
        self._attack_log: deque = deque(maxlen=200)  # 攻击记录
        self._attack_log_lock = threading.Lock()
        self._ts_buffer: deque = deque(maxlen=120)  # 时序数据（2分钟）
        self._ts_lock = threading.Lock()
        self._last_packets_captured = 0  # 用于差值计算
        # 增量统计计数器
        self._attack_type_counts: Dict[str, int] = {}
        self._attack_danger_counts: Dict[str, int] = {'严重': 0, '高危': 0, '中危': 0, '低危': 0}
        self._register_routes()

    def _register_routes(self):
        @self.app.route('/')
        def index():
            return render_template('dashboard.html')

        @self.app.route('/api/status')
        def api_status():
            with self._status_lock:
                return jsonify(self._status.to_dict())

        @self.app.route('/api/system')
        def api_system():
            info = get_system_info()
            return jsonify(info)

        @self.app.route('/api/resources')
        def api_resources():
            return jsonify({
                **get_resource_usage(),
                'timestamp': datetime.now().isoformat(),
            })

        @self.app.route('/api/stats')
        def api_stats():
            with self._status_lock:
                return jsonify({
                    'timestamp': datetime.now().isoformat(),
                    'status': self._status.to_dict(),
                    'system': get_system_info(),
                    'resources': get_resource_usage(),
                })

        @self.app.route('/api/control/start', methods=['POST'])
        def api_start():
            if 'start' in self._callbacks:
                try:
                    self._callbacks['start']()
                    return jsonify({'success': True, 'message': '检测已启动'})
                except Exception as e:
                    return jsonify({'success': False, 'error': str(e)}), 500
            return jsonify({'success': False, 'error': '未注册启动回调'}), 400

        @self.app.route('/api/control/stop', methods=['POST'])
        def api_stop():
            if 'stop' in self._callbacks:
                try:
                    self._callbacks['stop']()
                    return jsonify({'success': True, 'message': '检测已停止'})
                except Exception as e:
                    return jsonify({'success': False, 'error': str(e)}), 500
            return jsonify({'success': False, 'error': '未注册停止回调'}), 400

        @self.app.route('/api/timeseries')
        def api_timeseries():
            limit = request.args.get('limit', 120, type=int)
            with self._ts_lock:
                data = list(self._ts_buffer)[-limit:]
            return jsonify(data)

        @self.app.route('/api/attacks')
        def api_attacks():
            limit = request.args.get('limit', 50, type=int)
            with self._attack_log_lock:
                attacks = list(self._attack_log)[:limit]
                total = len(self._attack_log)
                type_counts = dict(self._attack_type_counts)
                danger_counts = dict(self._attack_danger_counts)
            return jsonify({
                'attacks': attacks,
                'total': total,
                'type_counts': type_counts,
                'danger_counts': danger_counts,
            })

    def register_callback(self, name: str, callback: Callable):
        self._callbacks[name] = callback

    def add_attack(self, record: dict):
        """添加攻击记录（增量更新统计计数器）"""
        with self._attack_log_lock:
            # 环形队列满时，淘汰最旧记录（最右端）并递减计数器
            if len(self._attack_log) == self._attack_log.maxlen:
                evicted = self._attack_log.pop()
                evicted_type = evicted.get('type', '未知')
                if evicted_type in self._attack_type_counts:
                    self._attack_type_counts[evicted_type] = max(
                        0, self._attack_type_counts[evicted_type] - 1
                    )
                evicted_danger = evicted.get('danger', '低危')
                if evicted_danger in self._attack_danger_counts:
                    self._attack_danger_counts[evicted_danger] = max(
                        0, self._attack_danger_counts[evicted_danger] - 1
                    )

            # 新记录插入队首
            self._attack_log.appendleft(record)

            # 增量更新计数器
            attack_type = record.get('type', '未知')
            self._attack_type_counts[attack_type] = (
                self._attack_type_counts.get(attack_type, 0) + 1
            )
            danger = record.get('danger', '低危')
            if danger in self._attack_danger_counts:
                self._attack_danger_counts[danger] += 1

    def update_status(self, **kwargs):
        with self._status_lock:
            for key, value in kwargs.items():
                if hasattr(self._status, key):
                    setattr(self._status, key, value)
        self._append_ts_snapshot()

    def _append_ts_snapshot(self):
        """追加一条时序快照到缓冲区"""
        try:
            res = get_resource_usage()
            with self._status_lock:
                s = self._status
                now = datetime.now().strftime('%H:%M:%S')
                # 每秒差值
                pkt_delta = s.packets_captured - self._last_packets_captured
                self._last_packets_captured = s.packets_captured
                if pkt_delta < 0:
                    pkt_delta = 0  # 计数器重置时

            snapshot = {
                'time': now,
                'packets_sec': pkt_delta,
                'flows_active': s.flows_active,
                'attacks_detected': s.attacks_detected,
                'attacks_total': s.attacks_total,
                'avg_latency_ms': round(s.avg_latency_ms, 2),
                'cpu_percent': round(res.get('cpu_percent', 0), 1),
                'memory_percent': round(res.get('memory_percent', 0), 1),
            }
            with self._ts_lock:
                self._ts_buffer.append(snapshot)
        except Exception:
            pass  # 静默处理，避免影响主流程

    def get_status(self) -> Dict[str, Any]:
        with self._status_lock:
            return self._status.to_dict()

    def run(self, threaded: bool = False):
        if threaded:
            t = threading.Thread(target=self._run, daemon=True)
            t.start()
        else:
            self._run()

    def _run(self):
        self.logger.info(f"仪表盘启动: http://{self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False)


def create_dashboard(config=None) -> DashboardServer:
    from pathlib import Path
    # Flask 的 template_folder 解析相对路径以 dashboard.py 所在位置为准
    # 这里用项目根目录的绝对路径
    project_root = Path(__file__).parent.parent.parent
    template_dir = str(project_root / 'templates')
    static_dir = str(project_root / 'static')
    if config and hasattr(config, 'web'):
        return DashboardServer(
            host=config.web.host, port=config.web.port, debug=config.web.debug,
            template_folder=template_dir, static_folder=static_dir,
        )
    return DashboardServer(template_folder=template_dir, static_folder=static_dir)
