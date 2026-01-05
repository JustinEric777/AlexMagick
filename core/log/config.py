import json
import logging
import logging.config
import os
import sys
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

import envs
from core.log.formatter import NewLineFormatter, ColoredFormatter


def _resolve_stream(stream_expr: str):
    """解析类似 ext://sys.stdout 的流配置。"""
    if stream_expr.startswith("ext://"):
        target = stream_expr.replace("ext://", "")
        module_name, attr_name = target.rsplit(".", 1)
        module = __import__(module_name, fromlist=[attr_name])
        return getattr(module, attr_name)
    return sys.stdout


def configure_logging() -> None:
    if envs.LOGGING_ENABLED == 0:
        return

    config_path = envs.LOGGING_CONFIG_PATH
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

    level_name = envs.LOGGING_LEVEL.upper()
    level = getattr(logging, level_name, logging.INFO)
    stream = _resolve_stream(envs.LOGGING_STREAM)

    root = logging.getLogger()
    root.setLevel(level)

    # 避免重复添加
    if root.handlers:
        return

    # Console Handler with Colored Output
    console = logging.StreamHandler(stream)
    console.setLevel(level)
    prefix = envs.LOGGING_PREFIX or ""
    # Format with relative path (fileinfo is populated by formatter)
    fmt = f"{prefix}[%(asctime)s] [%(levelname)s] %(name)s (%(fileinfo)s:%(lineno)d): %(message)s"
    formatter = ColoredFormatter(fmt)
    console.setFormatter(formatter)
    root.addHandler(console)


def init_logger(
        name: str,
        log_file: Optional[str] = None,
        level: Optional[int] = None,
        rotate_by_size: bool = False, # Changed default to favor time rotation per user req
        max_bytes: int = 5_000_000,
        backup_count: int = envs.LOGGING_RETENTION_DAYS,
        rotate_by_time: bool = True, # Changed default to True per user req
        when: str = "midnight",
        interval: int = 1,
) -> logging.Logger:
    """获取命名 logger，并根据 envs 完成默认控制台配置；可选增加文件输出。"""
    configure_logging()

    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)

    if log_file:
        # Ensure log directory exists
        log_path = Path(envs.LOGGING_DIR) / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file_str = str(log_path)

        # 避免重复添加同一路径的文件 handler
        exists = any(isinstance(h, (RotatingFileHandler, TimedRotatingFileHandler, logging.FileHandler)) and getattr(h, 'baseFilename', None) == str(Path(log_file_str).resolve())
                    for h in logger.handlers)
        
        if not exists:
            if rotate_by_time:
                fh = TimedRotatingFileHandler(log_file_str, when=when, interval=interval, backupCount=backup_count,
                                              encoding="utf-8")
                fh.suffix = "%Y-%m-%d"
            elif rotate_by_size:
                fh = RotatingFileHandler(log_file_str, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
            else:
                fh = logging.FileHandler(log_file_str, encoding="utf-8")

            # Plain text format for file
            fmt = f"{envs.LOGGING_PREFIX}[%(asctime)s] [%(levelname)s] %(name)s (%(fileinfo)s:%(lineno)d): %(message)s"
            fh.setFormatter(NewLineFormatter(fmt))
            logger.addHandler(fh)

    return logger
