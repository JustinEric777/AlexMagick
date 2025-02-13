import abc
from typing import Any


class BaseModel(metaclass=abc.ABCMeta):
    model: Any = None
    processor: Any = None
    device: Any = None
    dtype: Any = None

    @abc.abstractmethod
    def load_model(self, model_path: str, device: str):
        pass

    @abc.abstractmethod
    def generate(self, audio: str):
        pass

