from __future__ import annotations

import importlib
import inspect
from typing import Any, Iterable, List, Optional, Type


def import_module(module_path: str):
    return importlib.import_module(module_path)


def find_candidate_class(module, required_any_methods: List[str], class_suffix: Optional[str] = None) -> Type[Any]:
    """在模块中查找首个满足方法特征的类（可选按类名后缀过滤）。"""
    for _, cls in inspect.getmembers(module, inspect.isclass):
        if cls.__module__ != module.__name__:
            continue
        if class_suffix and not cls.__name__.endswith(class_suffix):
            continue
        methods = {m for m, f in inspect.getmembers(cls, inspect.isfunction)}
        if any(m in methods for m in required_any_methods):
            return cls
    raise ImportError(f"No class with methods {required_any_methods} found in {module.__name__}")


def call_first(obj: Any, method_candidates: List[str], *args, **kwargs):
    for name in method_candidates:
        if hasattr(obj, name):
            return getattr(obj, name)(*args, **kwargs)
    raise AttributeError(f"None of methods {method_candidates} found in {type(obj).__name__}")


