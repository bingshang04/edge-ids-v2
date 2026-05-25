"""
日志系统
"""
import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Optional

from ..config.constants import LOG_CONFIG, DEFAULT_LOG_DIR


class ColoredFormatter(logging.Formatter):
    """带颜色的控制台日志格式化器"""
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        original = record.levelname
        if sys.stdout.isatty():
            color = self.COLORS.get(record.levelname, '')
            record.levelname = f"{color}{record.levelname}{self.RESET}"
        result = super().format(record)
        record.levelname = original
        return result


def setup_logging(
    level: Optional[str] = None,
    log_dir: Optional[str] = None,
) -> logging.Logger:
    """配置日志系统"""
    level = level or LOG_CONFIG['level']
    log_dir = log_dir or DEFAULT_LOG_DIR
    log_level = getattr(logging, level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    fmt = LOG_CONFIG['format']
    datefmt = LOG_CONFIG['date_format']

    # 控制台输出（带颜色）
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    console.setFormatter(ColoredFormatter(fmt, datefmt=datefmt))
    root_logger.addHandler(console)

    # 文件输出
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_path / 'edge_ids.log',
        maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root_logger.addHandler(file_handler)

    # 错误日志单独文件
    error_handler = logging.handlers.RotatingFileHandler(
        log_path / 'edge_ids.error.log',
        maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root_logger.addHandler(error_handler)

    # 抑制第三方库日志
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('scapy').setLevel(logging.WARNING)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class LoggerMixin:
    """日志混入类"""
    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(
            self.__class__.__module__ + '.' + self.__class__.__name__
        )
