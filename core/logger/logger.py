import functools
from typing import Optional

from .config import init_logger


def get_logger(logger_name: Optional[str] = None, log_file: Optional[str] = None):
    """装饰器：为类或函数注入 `logger`。"""

    def decorator(obj):
        if isinstance(obj, type):
            orig_init = obj.__init__

            @functools.wraps(orig_init)
            def new_init(self, *args, **kwargs):
                name = logger_name or obj.__name__
                self.logger = init_logger(name=name, log_file=log_file)
                orig_init(self, *args, **kwargs)

            obj.__init__ = new_init
            return obj

        elif callable(obj):
            @functools.wraps(obj)
            def wrapper(*args, **kwargs):
                name = logger_name or obj.__module__
                logger = init_logger(name=name, log_file=log_file)
                kwargs['logger'] = logger
                return obj(*args, **kwargs)

            return wrapper
        else:
            raise TypeError("get_logger decorator can only be applied to classes or functions")

    return decorator