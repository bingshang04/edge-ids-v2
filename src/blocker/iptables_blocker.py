"""
IPTables 阻断器模块
实现基于置信度和攻击频率的三级阻断决策引擎
"""
import time
import threading
import subprocess
import re
import logging
import ipaddress
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional, List, Dict, Tuple

from ..utils.logger import LoggerMixin
from ..utils.platform_info import is_windows

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 自定义 iptables 链名
CHAIN_NAME = 'EDGE_IDS'

# 白名单网段（局域网）
DEFAULT_WHITELIST_CIDRS = [
    '192.168.0.0/16',
    '10.0.0.0/8',
    '172.16.0.0/12',
]

# 三级决策阈值
CONFIDENCE_HIGH = 0.95       # 高置信度阈值
CONFIDENCE_MEDIUM = 0.88     # 中置信度阈值
HIGH_WINDOW_SEC = 30         # 高置信度时间窗口（秒）
HIGH_COUNT_MIN = 3           # 高置信度窗口内最少攻击次数
MEDIUM_WINDOW_SEC = 60       # 中置信度时间窗口（秒）
MEDIUM_COUNT_MIN = 5         # 中置信度窗口内最少攻击次数
TEMP_BLOCK_DURATION = 1800   # 临时阻断时长（秒）= 30 分钟

# iptables 注释格式
COMMENT_TEMP_PREFIX = 'EdgeIDS:temp:'   # 临时阻断标记
COMMENT_PERM_PREFIX = 'EdgeIDS:perm'    # 永久阻断标记

# 清理线程扫描间隔
CLEANUP_INTERVAL_SEC = 30

# iptables 规则解析正则
RULE_LINE_RE = re.compile(
    r'-A\s+EDGE_IDS\s+-s\s+(\S+).*--comment\s+"([^"]+)"'
)


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

@dataclass
class BlockDecision:
    """阻断决策结果"""
    action: str                      # block_temp | block_permanent | whitelist_pass | log_only
    src_ip: str
    reason: str
    duration_sec: int = 0            # 阻断时长（秒），永久阻断为 0
    confidence: float = 0.0
    attack_type: str = 'Unknown'

    def is_blocked(self) -> bool:
        """是否执行了阻断"""
        return self.action in ('block_temp', 'block_permanent')


@dataclass
class BlockRecord:
    """阻断记录（内部追踪用）"""
    src_ip: str
    action: str
    expiry_ts: float = 0.0           # 到期时间戳（Unix 时间），0 表示永久
    created_at: float = field(default_factory=time.time)
    reason: str = ''
    attack_type: str = 'Unknown'

    @property
    def is_expired(self) -> bool:
        """是否已过期"""
        if self.expiry_ts <= 0:
            return False  # 永久阻断不过期
        return time.time() >= self.expiry_ts

    @property
    def is_permanent(self) -> bool:
        """是否永久阻断"""
        return self.action == 'block_permanent'


# ---------------------------------------------------------------------------
# 阻断器实现
# ---------------------------------------------------------------------------

class IPTablesBlocker(LoggerMixin):
    """
    iptables 阻断器

    三级决策引擎：
      1. 白名单 IP → 放行（-j RETURN）
      2. 置信度 >= 0.95 + 30s 内同 IP 攻击 >= 3 次 → 永久阻断
      3. 置信度 >= 0.88 + 60s 内同 IP 攻击 >= 5 次 → 临时阻断 30 分钟
      4. 其他 → 仅记录

    dry_run 模式：Windows 开发环境使用，仅记录日志不执行 iptables 命令
    """

    def __init__(
        self,
        dry_run: bool = False,
        whitelist_cidrs: Optional[List[str]] = None,
        chain_name: str = CHAIN_NAME,
        enable_cleanup: bool = True,
    ):
        super().__init__()

        self.dry_run = dry_run
        self.chain_name = chain_name
        self._enable_cleanup = enable_cleanup

        # 初始化白名单
        self._whitelist_networks: List[ipaddress.IPv4Network] = []
        for cidr in (whitelist_cidrs or DEFAULT_WHITELIST_CIDRS):
            try:
                self._whitelist_networks.append(
                    ipaddress.ip_network(cidr, strict=False)
                )
            except ValueError as e:
                self.logger.warning(f"无效的白名单 CIDR: {cidr}, 错误: {e}")

        # 攻击历史: IP → 攻击时间戳列表
        self._attack_history: Dict[str, List[float]] = defaultdict(list)
        self._history_lock = threading.Lock()

        # 已阻断 IP: IP → BlockRecord
        self._blocked_ips: Dict[str, BlockRecord] = {}
        self._blocked_lock = threading.Lock()

        # 初始化 iptables 自定义链
        self._init_chain()

        # 启动后台清理线程
        if enable_cleanup and not dry_run:
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_loop,
                daemon=True,
                name='iptables-cleanup',
            )
            self._cleanup_thread.start()
            self.logger.info('iptables 过期规则清理线程已启动')
        else:
            self._cleanup_thread = None

    # ------------------------------------------------------------------
    # iptables 命令封装
    # ------------------------------------------------------------------

    def _run_iptables(self, args: List[str], check_output: bool = False) -> Tuple[int, str]:
        """
        执行 iptables 命令

        Returns:
            (returncode, stdout_or_stderr)
        """
        cmd = ['iptables'] + args
        cmd_str = ' '.join(cmd)

        if self.dry_run:
            self.logger.info(f'[DRY_RUN] 将执行: {cmd_str}')
            return 0, ''

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                self.logger.error(f'iptables 命令失败: {cmd_str}\n{result.stderr.strip()}')
                return result.returncode, result.stderr.strip()
            output = result.stdout.strip()
            if check_output:
                return result.returncode, output
            return result.returncode, output
        except subprocess.TimeoutExpired:
            self.logger.error(f'iptables 命令超时: {cmd_str}')
            return -1, 'timeout'
        except FileNotFoundError:
            self.logger.error('iptables 命令不存在，请确认已安装 iptables')
            return -1, 'iptables not found'
        except Exception as e:
            self.logger.error(f'iptables 执行异常: {e}')
            return -1, str(e)

    # ------------------------------------------------------------------
    # 自定义链管理
    # ------------------------------------------------------------------

    def _init_chain(self):
        """初始化自定义 EDGE_IDS 链"""
        # 检查链是否存在
        rc, output = self._run_iptables(['-L', self.chain_name, '-n'], check_output=True)
        chain_exists = (rc == 0)

        if not chain_exists:
            # 创建新链
            rc, _ = self._run_iptables(['-N', self.chain_name])
            if rc != 0 and not self.dry_run:
                self.logger.error(f'创建 iptables 链 {self.chain_name} 失败')
                return

            # 将 EDGE_IDS 链挂载到 INPUT 链（在已有规则之前）
            self._run_iptables([
                '-I', 'INPUT', '1', '-j', self.chain_name,
                '-m', 'comment', '--comment', 'EdgeIDS:jump',
            ])
            self.logger.info(f'iptables 自定义链 {self.chain_name} 已创建并挂载到 INPUT')

            # 添加默认 RETURN 规则（链尾放行，确保不影响正常流量）
            self._run_iptables([
                '-A', self.chain_name, '-j', 'RETURN',
                '-m', 'comment', '--comment', 'EdgeIDS:default_return',
            ])
        else:
            self.logger.info(f'iptables 自定义链 {self.chain_name} 已存在')

        # 添加白名单规则（每次都检查，幂等操作会跳过已存在的规则）
        for network in self._whitelist_networks:
            self._run_iptables([
                '-I', self.chain_name, '1',
                '-s', str(network), '-j', 'RETURN',
                '-m', 'comment', '--comment', 'EdgeIDS:whitelist',
            ])

    def _block_ip(self, src_ip: str, duration_sec: int = 0, reason: str = ''):
        """
        通过 iptables 阻断指定 IP

        Args:
            src_ip: 源 IP 地址
            duration_sec: 阻断时长（秒），0 表示永久
            reason: 阻断原因
        """
        # 检查是否已阻断
        with self._blocked_lock:
            if src_ip in self._blocked_ips:
                existing = self._blocked_ips[src_ip]
                if existing.is_permanent:
                    self.logger.debug(f'IP {src_ip} 已被永久阻断，跳过')
                    return
                # 如果新的是永久阻断，升级
                if duration_sec == 0:
                    self._delete_rule(src_ip)
                else:
                    self.logger.debug(f'IP {src_ip} 已有临时阻断记录，跳过')
                    return

        # 构造注释
        if duration_sec > 0:
            expiry_ts = time.time() + duration_sec
            comment = f'{COMMENT_TEMP_PREFIX}{expiry_ts}'
        else:
            expiry_ts = 0.0
            comment = COMMENT_PERM_PREFIX

        # 添加 DROP 规则（插入到链首，优先级最高）
        rc, _ = self._run_iptables([
            '-I', self.chain_name, '1',
            '-s', src_ip, '-j', 'DROP',
            '-m', 'comment', '--comment', comment,
        ])
        if rc == 0:
            action = 'block_permanent' if duration_sec == 0 else 'block_temp'
            record = BlockRecord(
                src_ip=src_ip,
                action=action,
                expiry_ts=expiry_ts,
                reason=reason,
            )
            with self._blocked_lock:
                self._blocked_ips[src_ip] = record
            self.logger.info(
                f'已阻断 IP: {src_ip} | 类型: {action} | '
                f'持续: {duration_sec}s | 原因: {reason}'
            )

    def _delete_rule(self, src_ip: str):
        """删除指定 IP 的 iptables 规则"""
        rc, _ = self._run_iptables([
            '-D', self.chain_name, '-s', src_ip, '-j', 'DROP',
        ])
        if rc == 0:
            with self._blocked_lock:
                self._blocked_ips.pop(src_ip, None)
            self.logger.info(f'已解除 IP 阻断: {src_ip}')

    # ------------------------------------------------------------------
    # 白名单判断
    # ------------------------------------------------------------------

    def _is_whitelisted(self, src_ip: str) -> bool:
        """判断 IP 是否在白名单网段内"""
        try:
            addr = ipaddress.ip_address(src_ip)
            for network in self._whitelist_networks:
                if addr in network:
                    return True
        except ValueError:
            pass
        return False

    # ------------------------------------------------------------------
    # 攻击历史追踪
    # ------------------------------------------------------------------

    def _record_attack(self, src_ip: str):
        """记录一次攻击事件"""
        now = time.time()
        with self._history_lock:
            self._attack_history[src_ip].append(now)
            # 清理过期记录（保留最近 2 * MEDIUM_WINDOW_SEC 的数据）
            cutoff = now - 2 * MEDIUM_WINDOW_SEC
            self._attack_history[src_ip] = [
                ts for ts in self._attack_history[src_ip] if ts >= cutoff
            ]

    def _count_recent_attacks(self, src_ip: str, window_sec: float) -> int:
        """统计指定时间窗口内的攻击次数"""
        now = time.time()
        cutoff = now - window_sec
        with self._history_lock:
            history = self._attack_history.get(src_ip, [])
            return sum(1 for ts in history if ts >= cutoff)

    # ------------------------------------------------------------------
    # 三级决策引擎
    # ------------------------------------------------------------------

    def evaluate(
        self,
        src_ip: str,
        confidence: float,
        attack_type: str = 'Unknown',
    ) -> BlockDecision:
        """
        评估是否阻断源 IP

        决策逻辑（三级）：
          L1: 白名单 → 放行
          L2: 置信度 >= 0.95 + 30s 内 >= 3 次 → 永久阻断
          L3: 置信度 >= 0.88 + 60s 内 >= 5 次 → 临时阻断 30 分钟
          L4: 其他 → 仅记录

        Args:
            src_ip: 源 IP 地址
            confidence: 检测置信度
            attack_type: 攻击类型

        Returns:
            BlockDecision 决策结果
        """
        # L0: 检查是否已在阻断列表中
        with self._blocked_lock:
            if src_ip in self._blocked_ips:
                record = self._blocked_ips[src_ip]
                if not record.is_expired:
                    self.logger.debug(f'IP {src_ip} 已在阻断列表中，跳过评估')
                    return BlockDecision(
                        action='already_blocked',
                        src_ip=src_ip,
                        reason=f'已在阻断列表中 ({record.action})',
                        confidence=confidence,
                        attack_type=attack_type,
                    )

        # L1: 白名单检查
        if self._is_whitelisted(src_ip):
            self.logger.debug(f'IP {src_ip} 在白名单中，放行')
            return BlockDecision(
                action='whitelist_pass',
                src_ip=src_ip,
                reason='白名单放行',
                confidence=confidence,
                attack_type=attack_type,
            )

        # 记录本次攻击
        self._record_attack(src_ip)

        # L2: 高置信度 + 高频攻击 → 永久阻断
        if confidence >= CONFIDENCE_HIGH:
            count_30s = self._count_recent_attacks(src_ip, HIGH_WINDOW_SEC)
            if count_30s >= HIGH_COUNT_MIN:
                self._block_ip(
                    src_ip,
                    duration_sec=0,
                    reason=f'高置信度({confidence:.4f}) + {HIGH_WINDOW_SEC}s 内 {count_30s} 次攻击',
                )
                return BlockDecision(
                    action='block_permanent',
                    src_ip=src_ip,
                    reason=f'高置信度({confidence:.4f}) + {HIGH_WINDOW_SEC}s 内 {count_30s} 次攻击',
                    confidence=confidence,
                    attack_type=attack_type,
                )

        # L3: 中置信度 + 持续攻击 → 临时阻断
        if confidence >= CONFIDENCE_MEDIUM:
            count_60s = self._count_recent_attacks(src_ip, MEDIUM_WINDOW_SEC)
            if count_60s >= MEDIUM_COUNT_MIN:
                self._block_ip(
                    src_ip,
                    duration_sec=TEMP_BLOCK_DURATION,
                    reason=f'中置信度({confidence:.4f}) + {MEDIUM_WINDOW_SEC}s 内 {count_60s} 次攻击',
                )
                return BlockDecision(
                    action='block_temp',
                    src_ip=src_ip,
                    reason=f'中置信度({confidence:.4f}) + {MEDIUM_WINDOW_SEC}s 内 {count_60s} 次攻击',
                    duration_sec=TEMP_BLOCK_DURATION,
                    confidence=confidence,
                    attack_type=attack_type,
                )

        # L4: 其他 → 仅记录
        return BlockDecision(
            action='log_only',
            src_ip=src_ip,
            reason=f'未达到阻断阈值 (置信度: {confidence:.4f})',
            confidence=confidence,
            attack_type=attack_type,
        )

    # ------------------------------------------------------------------
    # 过期规则清理
    # ------------------------------------------------------------------

    def _scan_expired_rules(self) -> List[str]:
        """
        扫描 iptables 规则，找出已过期的临时阻断 IP

        Returns:
            已过期 IP 列表
        """
        rc, output = self._run_iptables(
            ['-L', self.chain_name, '-n', '--line-numbers'],
            check_output=True,
        )
        if rc != 0:
            return []

        expired_ips = []
        for line in output.split('\n'):
            match = RULE_LINE_RE.search(line)
            if not match:
                continue
            src_ip = match.group(1)
            comment = match.group(2)
            if comment.startswith(COMMENT_TEMP_PREFIX):
                try:
                    expiry_ts = float(comment[len(COMMENT_TEMP_PREFIX):])
                    if time.time() >= expiry_ts:
                        expired_ips.append(src_ip)
                except ValueError:
                    pass
        return expired_ips

    def _cleanup_expired(self):
        """清理所有过期的临时阻断规则"""
        expired_ips = self._scan_expired_rules()
        for ip in expired_ips:
            self._delete_rule(ip)
        if expired_ips:
            self.logger.info(f'已清理 {len(expired_ips)} 条过期阻断: {expired_ips}')

    def _cleanup_loop(self):
        """后台清理线程：每 30 秒扫描并删除过期规则"""
        while True:
            try:
                time.sleep(CLEANUP_INTERVAL_SEC)
                self._cleanup_expired()
            except Exception as e:
                self.logger.error(f'清理线程异常: {e}')

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    def get_blocked_ips(self) -> List[Dict]:
        """获取当前已阻断的 IP 列表"""
        with self._blocked_lock:
            return [
                {
                    'src_ip': ip,
                    'action': r.action,
                    'expiry_ts': r.expiry_ts,
                    'is_expired': r.is_expired,
                    'reason': r.reason,
                    'created_at': r.created_at,
                }
                for ip, r in self._blocked_ips.items()
            ]

    def get_stats(self) -> Dict:
        """获取阻断器统计信息"""
        with self._blocked_lock:
            block_count = len(self._blocked_ips)
            perm_count = sum(1 for r in self._blocked_ips.values() if r.is_permanent)
            temp_count = block_count - perm_count

        with self._history_lock:
            total_ips_tracked = len(self._attack_history)
            total_attacks = sum(len(v) for v in self._attack_history.values())

        return {
            'blocked_total': block_count,
            'blocked_permanent': perm_count,
            'blocked_temp': temp_count,
            'tracked_ips': total_ips_tracked,
            'total_attacks_recorded': total_attacks,
            'dry_run': self.dry_run,
        }

    def cleanup(self):
        """手动触发清理（退出前调用）"""
        self._cleanup_expired()

    def destroy(self):
        """
        销毁自定义链（清理所有规则）
        慎用：仅在完全停止系统时调用
        """
        if self.dry_run:
            self.logger.info('[DRY_RUN] 将销毁自定义链')
            return

        # 删除 INPUT 中的跳转规则
        self._run_iptables([
            '-D', 'INPUT', '-j', self.chain_name,
            '-m', 'comment', '--comment', 'EdgeIDS:jump',
        ])
        # 清空自定义链
        self._run_iptables(['-F', self.chain_name])
        # 删除自定义链
        self._run_iptables(['-X', self.chain_name])
        self.logger.info(f'iptables 自定义链 {self.chain_name} 已销毁')


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def create_blocker(config=None) -> IPTablesBlocker:
    """
    创建阻断器实例

    - Windows 平台：自动返回 dry_run 模式
    - Linux 非 root：返回 dry_run 模式（iptables 需要 root）
    - Linux root + 树莓派：返回正常模式

    Args:
        config: Settings 实例（可选）

    Returns:
        IPTablesBlocker 实例
    """
    dry_run = is_windows()

    # Linux 下检查 root 权限
    if not dry_run:
        import os
        try:
            if os.geteuid() != 0:
                logging.getLogger(__name__).warning(
                    '非 root 用户运行，iptables 阻断器将工作于 dry_run 模式'
                )
                dry_run = True
        except AttributeError:
            # Windows 下没有 geteuid
            dry_run = True

    return IPTablesBlocker(
        dry_run=dry_run,
        whitelist_cidrs=DEFAULT_WHITELIST_CIDRS,
    )
