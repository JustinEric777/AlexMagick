import abc
from typing import Any


class BaseModel(metaclass=abc.ABCMeta):
    model: Any = None
    tokenizer: Any = None
    device: Any = None

    @abc.abstractmethod
    def load_model(self, model_path: str, device: str):
        pass

    @abc.abstractmethod
    def text_encode(self, text: str):
        pass

    @abc.abstractmethod
    def video_encode(self, text: str):
        pass

