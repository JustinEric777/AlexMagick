from __future__ import annotations

from typing import Dict, Type, TypeVar

from .base import BaseModel
from .context import ModelContext

T = TypeVar("T", bound=BaseModel)


class ModelFactory:
    _registry: Dict[str, Type[T]] = {}

    @classmethod
    def register(cls, name: str):
        """装饰器：注册模型类到工厂注册表。"""

        def decorator(model_cls: Type[T]):
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, context: ModelContext) -> T:
        if context.model_type not in cls._registry:
            raise ValueError(f"Unknown model type: {context.model_type}")
        model_cls = cls._registry[context.model_type]
        model = model_cls(
            model_name_or_path=context.model_name_or_path,
            device=context.device,
            **(context.extra_options or {}),
        )
        model.load()
        return model
