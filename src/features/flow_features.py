"""
48维流特征提取器（与训练脚本特征工程完全一致）

特征顺序（与 train.py load_and_preprocess() 输出的 feature_cols 严格一致）:
- 数值特征 [0-38] (39维): 按 UNSW-NB15 CSV 列自然顺序排列
- 衍生特征 [39-44] (6维): byte_ratio, load_ratio, pkt_ratio, dur_rate, ttl_diff, avg_pkt_size
- 类别特征 [45-47] (3维): proto, service, state

重要: 顺序必须与训练端完全一致，否则推理结果无意义。
"""
import time
import numpy as np
import joblib
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
from pathlib import Path

from ..utils.logger import LoggerMixin
from ..utils.helpers import safe_divide, get_flow_id
from ..capture.packet_capture import PacketInfo

FEATURE_DIM = 48

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent.parent.resolve()


@dataclass
class FlowStats:
    """单条流统计"""
    flow_id: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    start_time: float
    last_time: float

    fwd_packets: int = 0
    bwd_packets: int = 0
    fwd_bytes: int = 0
    bwd_bytes: int = 0

    packet_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    fwd_packet_times: deque = field(default_factory=lambda: deque(maxlen=500))
    bwd_packet_times: deque = field(default_factory=lambda: deque(maxlen=500))
    fwd_packet_lengths: deque = field(default_factory=lambda: deque(maxlen=500))
    bwd_packet_lengths: deque = field(default_factory=lambda: deque(maxlen=500))
    fwd_iat_list: deque = field(default_factory=lambda: deque(maxlen=100))
    bwd_iat_list: deque = field(default_factory=lambda: deque(maxlen=100))

    fin_flags: int = 0
    syn_flags: int = 0
    rst_flags: int = 0
    psh_flags: int = 0
    ack_flags: int = 0
    urg_flags: int = 0

    fwd_header_bytes: int = 0
    bwd_header_bytes: int = 0

    def update(self, pkt: PacketInfo) -> None:
        ts, length, direction, flags = pkt.timestamp, pkt.length, pkt.direction, pkt.flags
        self.last_time = ts
        self.packet_times.append(ts)

        header_len = 40 if self.protocol == "TCP" else 28

        if direction == "fwd":
            if self.fwd_packet_times:
                self.fwd_iat_list.append(ts - self.fwd_packet_times[-1])
            self.fwd_packets += 1
            self.fwd_bytes += length
            self.fwd_packet_times.append(ts)
            self.fwd_packet_lengths.append(length)
            self.fwd_header_bytes += header_len
        else:
            if self.bwd_packet_times:
                self.bwd_iat_list.append(ts - self.bwd_packet_times[-1])
            self.bwd_packets += 1
            self.bwd_bytes += length
            self.bwd_packet_times.append(ts)
            self.bwd_packet_lengths.append(length)
            self.bwd_header_bytes += header_len

        for char, attr in [('F', 'fin_flags'), ('S', 'syn_flags'), ('R', 'rst_flags'),
                           ('P', 'psh_flags'), ('A', 'ack_flags'), ('U', 'urg_flags')]:
            if char in flags:
                setattr(self, attr, getattr(self, attr) + 1)

    @property
    def duration(self) -> float:
        return max(self.last_time - self.start_time, 0.0001)

    @property
    def total_packets(self) -> int:
        return self.fwd_packets + self.bwd_packets

    @property
    def total_bytes(self) -> int:
        return self.fwd_bytes + self.bwd_bytes

    def is_expired(self, current_time: float, timeout: float) -> bool:
        return (current_time - self.last_time) > timeout

    def is_finished(self) -> bool:
        return self.fin_flags > 0 or self.rst_flags > 0


class FeatureExtractor(LoggerMixin):
    """48维流特征提取器 — 无法实时提取的特征用训练集均值填充，避免误报"""

    # ===== 特征索引常量（顺序与训练端 train.py 的 feature_cols 严格一致）=====
    #
    # 训练端顺序 = numeric_cols(CSV自然顺序) + 6衍生 + 3类别
    # UNSW-NB15 CSV 列顺序（排除 id, proto, service, state, attack_cat, label）:
    #   dur, spkts, dpkts, sbytes, dbytes, rate, sttl, dttl, sload, dload,
    #   sloss, dloss, sinpkt, dinpkt, sjit, djit, swin, stcpb, dtcpb, dwin,
    #   tcprtt, synack, ackdat, smean, dmean, trans_depth, response_body_len,
    #   ct_srv_src, ct_state_ttl, ct_dst_ltm, ct_src_dport_ltm, ct_dst_sport_ltm,
    #   ct_dst_src_ltm, is_ftp_login, ct_ftp_cmd, ct_flw_http_mthd, ct_src_ltm,
    #   ct_srv_dst, is_sm_ips_ports
    #   共 39 个数值特征

    # 数值特征 [0-38] — 按 CSV 列自然顺序
    IDX_DUR = 0
    IDX_SPKTS = 1
    IDX_DPKTS = 2
    IDX_SBYTES = 3
    IDX_DBYTES = 4
    IDX_RATE = 5
    IDX_STTL = 6
    IDX_DTTL = 7
    IDX_SLOAD = 8
    IDX_DLOAD = 9
    IDX_SLOSS = 10
    IDX_DLOSS = 11
    IDX_SINPKT = 12
    IDX_DINPKT = 13
    IDX_SJIT = 14
    IDX_DJIT = 15
    IDX_SWIN = 16
    IDX_STCPB = 17
    IDX_DTCPB = 18
    IDX_DWIN = 19
    IDX_TCPRTT = 20
    IDX_SYNACK = 21
    IDX_ACKDAT = 22
    IDX_SMEAN = 23
    IDX_DMEAN = 24
    IDX_TRANS_DEPTH = 25
    IDX_RESPONSE_BODY_LEN = 26
    IDX_CT_SRV_SRC = 27
    IDX_CT_STATE_TTL = 28
    IDX_CT_DST_LTM = 29
    IDX_CT_SRC_DPORT_LTM = 30
    IDX_CT_DST_SPORT_LTM = 31
    IDX_CT_DST_SRC_LTM = 32
    IDX_IS_FTP_LOGIN = 33
    IDX_CT_FTP_CMD = 34
    IDX_CT_FLW_HTTP_MTHD = 35
    IDX_CT_SRC_LTM = 36
    IDX_CT_SRV_DST = 37
    IDX_IS_SM_IPS_PORTS = 38

    # 衍生特征 [39-44]
    IDX_BYTE_RATIO = 39
    IDX_LOAD_RATIO = 40
    IDX_PKT_RATIO = 41
    IDX_DUR_RATE = 42
    IDX_TTL_DIFF = 43
    IDX_AVG_PKT_SIZE = 44

    # 类别特征 [45-47]
    IDX_PROTO = 45
    IDX_SERVICE = 46
    IDX_STATE = 47

    def __init__(self, flow_timeout: float = 120.0, max_flows: int = 10000,
                 scaler_path: Optional[str] = None,
                 le_proto_path: Optional[str] = None,
                 le_service_path: Optional[str] = None,
                 le_state_path: Optional[str] = None):
        super().__init__()
        self.flow_timeout = flow_timeout
        self.max_flows = max_flows
        self._flows: Dict[str, FlowStats] = {}
        self._last_cleanup = time.time()
        self._stats = {'flows_created': 0, 'flows_completed': 0, 'flows_expired': 0, 'packets_processed': 0}

        # 加载 Scaler 的 mean_ 作为不可提取特征的默认值
        self._defaults = np.zeros(48, dtype=np.float32)
        self._defaults[self.IDX_STTL] = 254.0
        self._defaults[self.IDX_DTTL] = 252.0
        if scaler_path:
            self._load_scaler_defaults(scaler_path)
        else:
            # 兜底：用 UNSW-NB15 典型正常流量均值
            self._defaults[self.IDX_SWIN] = 255.0
            self._defaults[self.IDX_DWIN] = 255.0

        # 加载 LabelEncoder（用于 proto/service/state 编码，与训练端一致）
        self._le_proto = self._load_label_encoder(le_proto_path) if le_proto_path else None
        self._le_service = self._load_label_encoder(le_service_path) if le_service_path else None
        self._le_state = self._load_label_encoder(le_state_path) if le_state_path else None

        if self._le_proto:
            self.logger.info(f'已加载 Proto LabelEncoder: {len(self._le_proto.classes_)} 类')
        if self._le_service:
            self.logger.info(f'已加载 Service LabelEncoder: {len(self._le_service.classes_)} 类')
        if self._le_state:
            self.logger.info(f'已加载 State LabelEncoder: {len(self._le_state.classes_)} 类')

    @staticmethod
    def _load_label_encoder(path: str):
        """加载 LabelEncoder，失败返回 None"""
        try:
            if Path(path).exists():
                return joblib.load(path)
        except Exception as e:
            pass
        return None

    def _encode_proto(self, proto: str) -> int:
        """使用训练端 LabelEncoder 编码协议，未见过的协议做最佳猜测映射"""
        if self._le_proto is None:
            # 降级：硬编码映射（仅限常见协议）
            proto_lower = proto.lower().strip()
            hardcoded = {'tcp': 0, 'udp': 1, 'icmp': 2}
            return hardcoded.get(proto_lower, 0)

        proto_clean = proto.lower().strip() if proto else 'unknown'

        # 尝试精确匹配
        classes_lower = {c.lower(): i for i, c in enumerate(self._le_proto.classes_)}
        if proto_clean in classes_lower:
            return classes_lower[proto_clean]

        # 最佳猜测映射（常见缩写）
        fallback_map = {
            'tcp': 'tcp', 'udp': 'udp', 'icmp': 'icmp', 'igmp': 'igmp',
            'arp': 'arp', 'ip': 'ip', 'ipv6': 'ipv6', 'rtp': 'rtp',
        }
        mapped = fallback_map.get(proto_clean, 'unknown')
        if mapped in classes_lower:
            return classes_lower[mapped]

        # 最终降级：'unknown' 的编码
        if 'unknown' in classes_lower:
            return classes_lower['unknown']
        return 0

    def _encode_service(self, service: str) -> int:
        """使用训练端 LabelEncoder 编码服务类型"""
        if self._le_service is None:
            return 0
        service_clean = service.lower().strip() if service else '-'
        if service_clean == '' or service_clean is None:
            service_clean = '-'
        classes_lower = {c.lower(): i for i, c in enumerate(self._le_service.classes_)}
        if service_clean in classes_lower:
            return classes_lower[service_clean]
        # 降级到 '-'
        if '-' in classes_lower:
            return classes_lower['-']
        return 0

    def _encode_state(self, state: str) -> int:
        """使用训练端 LabelEncoder 编码连接状态"""
        if self._le_state is None:
            return 0
        state_clean = state.upper().strip() if state else 'FIN'
        if state_clean == '' or state_clean is None:
            state_clean = 'FIN'
        classes_upper = {c.upper(): i for i, c in enumerate(self._le_state.classes_)}
        if state_clean in classes_upper:
            return classes_upper[state_clean]
        # 降级到 'FIN'
        if 'FIN' in classes_upper:
            return classes_upper['FIN']
        return 0

    def _load_scaler_defaults(self, scaler_path: str):
        """从训练 Scaler 中提取 mean_ 作为缺失特征默认值"""
        from pathlib import Path
        if Path(scaler_path).exists():
            try:
                import joblib
                scaler = joblib.load(scaler_path)
                if hasattr(scaler, 'mean_'):
                    self._defaults = scaler.mean_.copy().astype(np.float32)
                    self.logger.info(f'已加载 Scaler 均值 → {scaler_path} ({len(self._defaults)}维)')
            except Exception as e:
                self.logger.warning(f'加载 Scaler 默认值失败: {e}，使用内置默认值')

    @property
    def active_flow_count(self) -> int:
        return len(self._flows)

    def _get_or_create_flow(self, pkt: PacketInfo) -> FlowStats:
        if pkt.flow_id not in self._flows:
            if len(self._flows) >= self.max_flows:
                oldest = min(self._flows.keys(), key=lambda k: self._flows[k].last_time)
                del self._flows[oldest]
            self._flows[pkt.flow_id] = FlowStats(
                flow_id=pkt.flow_id, src_ip=pkt.src_ip, dst_ip=pkt.dst_ip,
                src_port=pkt.src_port, dst_port=pkt.dst_port,
                protocol=pkt.protocol, start_time=pkt.timestamp, last_time=pkt.timestamp,
            )
            self._stats['flows_created'] += 1
        return self._flows[pkt.flow_id]

    def _cleanup(self, current_time: float) -> int:
        expired = [fid for fid, f in self._flows.items() if f.is_expired(current_time, self.flow_timeout)]
        for fid in expired:
            del self._flows[fid]
        self._stats['flows_expired'] += len(expired)
        self._last_cleanup = current_time
        return len(expired)

    @staticmethod
    def _calc_stats(data: List[float]):
        if not data:
            return 0.0, 0.0, 0.0, 0.0
        arr = np.array(data, dtype=np.float32)
        return float(arr.min()), float(arr.max()), float(arr.mean()), float(arr.std()) if len(arr) > 1 else 0.0

    def _extract_48_features(self, flow: FlowStats) -> np.ndarray:
        """
        提取48维特征（顺序与 train.py 的 feature_cols 严格一致）

        核心策略：以训练集 Scaler 均值为底（z≈0，对预测中性），
        只覆盖实际能从数据包中提取的特征，避免硬编码 0 导致误报。
        """
        dur = flow.duration
        tp = flow.total_packets
        sbytes_val = float(flow.fwd_bytes)
        dbytes_val = float(flow.bwd_bytes)

        # 从 Scaler 均值开始
        f = self._defaults.copy()

        # === 数值特征 [0-38] — 按 CSV 列自然顺序覆盖 ===
        # 0: dur
        f[self.IDX_DUR] = dur
        # 1: spkts (fwd_packets)
        f[self.IDX_SPKTS] = float(flow.fwd_packets)
        # 2: dpkts (bwd_packets)
        f[self.IDX_DPKTS] = float(flow.bwd_packets)
        # 3: sbytes
        f[self.IDX_SBYTES] = sbytes_val
        # 4: dbytes
        f[self.IDX_DBYTES] = dbytes_val
        # 5: rate
        f[self.IDX_RATE] = safe_divide(tp, dur)
        # 6-7: sttl, dttl — 保留 Scaler 均值（无法实时提取 TTL）
        # 8-11: sload, dload, sloss, dloss — 保留 Scaler 均值
        # 12-13: sinpkt, dinpkt — 用 IAT 统计近似
        fwd_iats = list(flow.fwd_iat_list)
        bwd_iats = list(flow.bwd_iat_list)
        f[self.IDX_SINPKT] = self._calc_stats(fwd_iats)[2] if fwd_iats else f[self.IDX_SINPKT]
        f[self.IDX_DINPKT] = self._calc_stats(bwd_iats)[2] if bwd_iats else f[self.IDX_DINPKT]
        # 14: sjit — 前向 IAT 均值
        f[self.IDX_SJIT] = self._calc_stats(fwd_iats)[2] if fwd_iats else f[self.IDX_SJIT]
        # 15: djit — 后向 IAT 均值
        f[self.IDX_DJIT] = self._calc_stats(bwd_iats)[2] if bwd_iats else f[self.IDX_DJIT]
        # 16-18: swin, stcpb, dtcpb — 保留 Scaler 均值
        # 19: dwin — 保留 Scaler 均值
        # 20: tcprtt — 全部 IAT 统计
        all_iats = fwd_iats + bwd_iats
        f[self.IDX_TCPRTT] = self._calc_stats(all_iats)[2] if all_iats else f[self.IDX_TCPRTT]
        # 21: synack — SYN 标志计数
        f[self.IDX_SYNACK] = float(flow.syn_flags)
        # 22: ackdat — ACK 标志计数
        f[self.IDX_ACKDAT] = float(flow.ack_flags)
        # 23-24: smean, dmean — 包大小均值（对应训练端的 smeansz/dmeansz）
        f[self.IDX_SMEAN] = safe_divide(sbytes_val, float(flow.fwd_packets))
        f[self.IDX_DMEAN] = safe_divide(dbytes_val, float(flow.bwd_packets))
        # 25-38: trans_depth, response_body_len, ct_* 系列 — 保留 Scaler 均值

        # === 衍生特征 [39-44] — 基于可提取特征重新计算 ===
        f[self.IDX_BYTE_RATIO] = safe_divide(sbytes_val, dbytes_val)
        # load_ratio — 保留 Scaler 均值（无法提取 sload/dload）
        f[self.IDX_PKT_RATIO] = safe_divide(float(flow.fwd_packets), float(tp))
        f[self.IDX_DUR_RATE] = dur * safe_divide(tp, dur)
        # ttl_diff: 如果有真实 TTL 值则计算，否则保留 Scaler 均值
        if f[self.IDX_STTL] > 0 and f[self.IDX_DTTL] > 0:
            f[self.IDX_TTL_DIFF] = f[self.IDX_STTL] - f[self.IDX_DTTL]
        f[self.IDX_AVG_PKT_SIZE] = safe_divide(sbytes_val + dbytes_val, float(tp))

        # === 类别特征 [45-47] — 使用训练端 LabelEncoder ===
        f[self.IDX_PROTO] = float(self._encode_proto(flow.protocol))
        f[self.IDX_SERVICE] = float(self._encode_service('-'))   # 推理端无法实时获取 service
        f[self.IDX_STATE] = float(self._encode_state('FIN'))     # 流结束时状态

        assert len(f) == 48, f'特征维度错误: {len(f)}'
        return np.array(f, dtype=np.float32)

    def process_packet(self, pkt: PacketInfo) -> Optional[np.ndarray]:
        t = time.time()
        if t - self._last_cleanup > 10.0:
            self._cleanup(t)

        flow = self._get_or_create_flow(pkt)
        flow.update(pkt)
        self._stats['packets_processed'] += 1

        if flow.is_finished():
            features = self._extract_48_features(flow)
            # 附加流统计到 packet_info，用于后续规则过滤
            pkt.flow_ack_flags = flow.ack_flags
            pkt.flow_dpkts = flow.bwd_packets
            pkt.flow_fin_flags = flow.fin_flags
            del self._flows[pkt.flow_id]
            self._stats['flows_completed'] += 1
            return features
        return None

    def force_extract_all(self) -> List[np.ndarray]:
        results = [self._extract_48_features(f) for f in list(self._flows.values())]
        self._flows.clear()
        return results

    def get_stats(self) -> Dict[str, Any]:
        return dict(self._stats)

    def clear(self) -> int:
        count = len(self._flows)
        self._flows.clear()
        return count


def create_feature_extractor(config=None) -> FeatureExtractor:
    scaler_path = None
    le_proto_path = None
    le_service_path = None
    le_state_path = None

    if config and hasattr(config, 'inference'):
        project_root = Path(__file__).parent.parent.parent
        if config.inference.scaler_path:
            scaler_path = str(project_root / config.inference.scaler_path)
        # LabelEncoder 路径（默认在 data/models/ 下）
        model_dir = project_root / 'data' / 'models'
        le_proto_path = str(model_dir / 'le_proto.joblib')
        le_service_path = str(model_dir / 'le_service.joblib')
        le_state_path = str(model_dir / 'le_state.joblib')

    kwargs = dict(scaler_path=scaler_path,
                  le_proto_path=le_proto_path,
                  le_service_path=le_service_path,
                  le_state_path=le_state_path)
    if config and hasattr(config, 'feature'):
        kwargs.update(flow_timeout=config.feature.flow_timeout,
                      max_flows=config.feature.max_flows)

    return FeatureExtractor(**kwargs)
