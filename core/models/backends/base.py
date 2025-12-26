from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Iterator


class BackendRunner(ABC):
    """后端执行器抽象：不同推理后端实现该接口以复用任务层逻辑。"""

    def __init__(self, device: str = "cpu", **kwargs: Any) -> None:
        self.device = device
        self.options = kwargs

    @abstractmethod
    def load_text_generation(self, model_name_or_path: str, **kwargs: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def generate_text(self, input_ids, **gen_kwargs: Any):
        """底层文本生成。输入为框架张量，由任务层负责准备。"""
        raise NotImplementedError

    def stream_chat(self, messages: list[dict], **kwargs) -> Iterator[str]:
        """
        统一的流式对话接口。
        默认实现为抛出未实现异常，子类应按需实现。
        返回的是生成的文本片段。
        """
        raise NotImplementedError("stream_chat is not implemented for this backend")


