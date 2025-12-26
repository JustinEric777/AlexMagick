import json
import logging
import logging.config
import os
import sys
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

from .formatter import NewLineFormatter


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[41m",
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


def _resolve_stream(stream_expr: str):
    """解析类似 ext://sys.stdout 的流配置。"""
    if stream_expr.startswith("ext://"):
        target = stream_expr.replace("ext://", "")
        module_name, attr_name = target.rsplit(".", 1)
        module = __import__(module_name, fromlist=[attr_name])
        return getattr(module, attr_name)
    return sys.stdout


def configure_logging() -> None:
    if int(os.getenv("VLLM_CONFIGURE_LOGGING", "1")) == 0:
        return

    config_path = os.getenv("VLLM_LOGGING_CONFIG_PATH")
    if config_path:
        path = Path(config_path)
        if path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                logging.config.dictConfig(config)
                return
            except Exception:
                # 回退到默认配置
                pass

    level_name = os.getenv("VLLM_LOGGING_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    stream = _resolve_stream(os.getenv("VLLM_LOGGING_STREAM", "ext://sys.stdout"))

    root = logging.getLogger()
    root.setLevel(level)

    # 避免重复添加
    if root.handlers:
        return

    console = logging.StreamHandler(stream)
    console.setLevel(level)

    prefix = os.getenv("VLLM_LOGGING_PREFIX", "") or ""
    fmt = f"{prefix}[%(asctime)s] [%(levelname)s] %(name)s (%(fileinfo)s:%(lineno)d): %(message)s"
    formatter = NewLineFormatter(fmt)

    console.setFormatter(formatter)
    root.addHandler(console)


def init_logger(
        name: str,
        log_file: Optional[str] = None,
        level: Optional[int] = None,
        rotate_by_size: bool = True,
        max_bytes: int = 5_000_000,
        backup_count: int = 5,
        rotate_by_time: bool = False,
        when: str = "midnight",
        interval: int = 1,
) -> logging.Logger:
    """获取命名 logger，并根据 envs 完成默认控制台配置；可选增加文件输出。"""
    configure_logging()

    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)

    if log_file:
        # 避免重复添加同一路径的文件 handler
        exists = any(isinstance(h, (RotatingFileHandler, TimedRotatingFileHandler, logging.FileHandler)) and getattr(h, 'baseFilename', None) == log_file  # type: ignore[attr-defined]
                    for h in logger.handlers)
        if not exists:
            if rotate_by_size:
                fh = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
            elif rotate_by_time:
                fh = TimedRotatingFileHandler(log_file, when=when, interval=interval, backupCount=backup_count,
                                              encoding="utf-8")
                fh.suffix = "%Y-%m-%d"
            else:
                fh = logging.FileHandler(log_file, encoding="utf-8")

            fmt = f"{os.getenv('VLLM_LOGGING_PREFIX', '')}[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
            fh.setFormatter(logging.Formatter(fmt))
            logger.addHandler(fh)

    return logger
