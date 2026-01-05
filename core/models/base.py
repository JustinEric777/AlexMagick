from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional


class BaseModel(ABC):
    """抽象基类：统一模型生命周期与推理接口
    最小约束：
    - 实现 `load()` 完成权重与资源初始化
    - 实现 `generate()` 文本生成/推理入口（可按需覆盖为具体任务）
    - 可选实现 `close()` 释放资源
    """

    def __init__(self, model_name_or_path: Optional[str] = None, device: str = "cpu", **kwargs: Any) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.extra_options = kwargs

    @abstractmethod
    def load(self) -> None:
        """加载/初始化模型与相关资源。"""
        raise NotImplementedError

    @abstractmethod
    def generate(self, inputs: Iterable[str] | str, **kwargs: Any) -> Any:
        """核心推理接口：文本生成/问答/通用推理。

        推荐接受批量 inputs（Iterable[str]）或单条 str，并返回与之对应的输出结构。
        """
        raise NotImplementedError

    def close(self) -> None:
        """释放资源（可选）。"""
        return
