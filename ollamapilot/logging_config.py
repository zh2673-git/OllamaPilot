"""
OllamaPilot 日志配置模块

提供统一的日志配置，支持：
- 控制台输出（带颜色）
- 文件输出
- 按模块设置日志级别
"""

import logging
import sys
from pathlib import Path
from typing import Optional


DEFAULT_FORMAT = "%(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器"""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
) -> logging.Logger:
    """
    设置项目日志配置

    Args:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径 (可选)
        log_format: 日志格式 (可选)
        date_format: 日期格式 (可选)

    Returns:
        根日志记录器
    """
    fmt = log_format or DEFAULT_FORMAT
    date_fmt = date_format or DEFAULT_DATE_FORMAT

    root_logger = logging.getLogger("ollamapilot")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if root_logger.handlers:
        root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_formatter = ColoredFormatter(fmt, datefmt=date_fmt)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(fmt, datefmt=date_fmt)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    获取模块专属日志记录器

    Args:
        name: 模块名称，通常使用 __name__

    Returns:
        日志记录器实例
    """
    return logging.getLogger(f"ollamapilot.{name}")


def set_module_level(module_name: str, level: str):
    """
    设置特定模块的日志级别

    Args:
        module_name: 模块名称 (如 "agent", "skills.graphrag")
        level: 日志级别
    """
    logger = logging.getLogger(f"ollamapilot.{module_name}")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))


_default_logger: Optional[logging.Logger] = None


def get_default_logger() -> logging.Logger:
    """获取默认日志记录器（懒加载）"""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logging()
    return _default_logger
