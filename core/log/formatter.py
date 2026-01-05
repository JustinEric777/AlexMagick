import logging
import os
import site
import sys
from pathlib import Path


class NewLineFormatter(logging.Formatter):
    """
    Plain text formatter that handles relative paths and newlines.
    """

    def __init__(self, fmt, datefmt=None, style="%"):
        super().__init__(fmt, datefmt, style)
        # Always use relative path logic as per requirement
        self.root_dir = Path(os.getcwd()).resolve()
        
        # Determine site-packages paths for third-party libraries
        self.site_packages = []
        if hasattr(site, 'getsitepackages'):
            self.site_packages.extend([Path(p).resolve() for p in site.getsitepackages()])
        if hasattr(site, 'getusersitepackages'):
            self.site_packages.append(Path(site.getusersitepackages()).resolve())

    def _get_rel_path(self, abs_path_str: str) -> str:
        if not abs_path_str:
            return ""
            
        path = Path(abs_path_str).resolve()
        
        # 1. Try project root relative
        try:
            return str(path.relative_to(self.root_dir))
        except ValueError:
            pass
            
        # 2. Try site-packages relative (third-party)
        for sp in self.site_packages:
            try:
                rel = path.relative_to(sp)
                return str(rel)
            except ValueError:
                continue
                
        # 3. Fallback to filename or absolute path if not in project or site-packages
        return path.name

    def _prepare_record(self, record):
        """Prepare custom fields in record like fileinfo."""
        abs_path = getattr(record, "pathname", None)
        if abs_path:
            record.fileinfo = self._get_rel_path(abs_path)
        else:
            record.fileinfo = record.filename

    def format(self, record):
        self._prepare_record(record)

        msg = super().format(record)
        if record.message != "":
            parts = msg.split(record.message)
            if len(parts) > 1:
                # Add indentation for multiline messages
                msg = msg.replace("\n", "\r\n" + parts[0])
        return msg


class ColoredFormatter(NewLineFormatter):
    """
    Formatter that adds colors to specific fields following the requested scheme:
    Timestamp (Red) Level (Dynamic) File (Plain) Func/Name (Cyan) Message (Yellow)
    """
    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "CRITICAL": "\033[41m", # Red background
    }
    RESET = "\033[0m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"

    def format(self, record):
        self._prepare_record(record)
        
        # Format Timestamp
        record.asctime = self.formatTime(record, self.datefmt)
        
        # Format Level
        level_color = self.COLORS.get(record.levelname, self.RESET)
        
        # Format Message (with indentation support if needed, but for colored output 
        # usually we just dump it, or we try to respect the indentation)
        # Note: self.formatMessage(record) uses self._fmt which might not be what we want if we are constructing manually.
        # But we can just use record.getMessage()
        
        msg = record.getMessage()
        
        # If we want multiline indentation matching the prefix length, it's tricky with manual construction.
        # But let's try to match the style first.
        
        # Structure: Red([Timestamp]) LevelColor([Level]) File:Line Cyan(Name) Yellow(Message)
        
        # 1. Timestamp
        timestamp_str = f"{self.RED}[{record.asctime}]{self.RESET}"
        
        # 2. Level
        level_str = f"{level_color}[{record.levelname}]{self.RESET}"
        
        # 3. File info (Plain)
        file_str = f"{record.fileinfo}:{record.lineno}"
        
        # 4. Name/Func (Cyan)
        name_str = f"{self.CYAN}{record.name}{self.RESET}"
        
        # 5. Message (Yellow)
        # Note: We apply yellow to the whole message.
        # Handling newlines: if we want them colored, we wrap the whole thing.
        msg_str = f"{self.YELLOW}{msg}{self.RESET}"
        
        # Indentation logic
        # We construct the header first to know the length? 
        # Or simply join them.
        
        # Example output:
        # [2023-01-01 10:00:00] [INFO] core/log/config.py:50 root This is message
        
        header = f"{timestamp_str} {level_str} {file_str} {name_str} "
        
        # If message has newlines, indent them?
        if "\n" in msg_str:
             # Calculate plain text length of header for indentation?
             # It's hard due to ANSI codes. 
             # Let's just indent with a fixed tab or space for now, or skip indentation to keep it simple 
             # as the user asked for specific coloring, not indentation.
             # But NewLineFormatter had indentation.
             pass
        
        return f"{header}{msg_str}"
