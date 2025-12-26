import logging
import os
from pathlib import Path


class NewLineFormatter(logging.Formatter):
    """为多行日志的每一行添加统一前缀，便于对齐阅读。"""

    def __init__(self, fmt, datefmt=None, style="%"):
        super().__init__(fmt, datefmt, style)
        self.use_relpath = os.getenv("VLLM_LOGGING_LEVEL", "INFO").upper() == "DEBUG"
        if self.use_relpath:
            # 工程根目录：core/ 之上的项目根
            self.root_dir = Path(__file__).resolve().parents[2]

    def format(self, record):

        def shrink_path(relpath: Path) -> str:
            parts = list(relpath.parts)
            new_parts = []
            if parts:
                new_parts += parts[:1]
                parts = parts[1:]
            if len(parts) > 2:
                new_parts += ["..."] + parts[-2:]
            else:
                new_parts += parts
            return "/".join(new_parts)

        if self.use_relpath:
            abs_path = getattr(record, "pathname", None)
            if abs_path:
                try:
                    relpath = Path(abs_path).resolve().relative_to(self.root_dir)
                except Exception:
                    relpath = Path(record.filename)
            else:
                relpath = Path(record.filename)
            record.fileinfo = shrink_path(relpath)
        else:
            record.fileinfo = record.filename

        msg = super().format(record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg
