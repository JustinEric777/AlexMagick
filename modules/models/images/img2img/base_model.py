import abc
from typing import Any


class BaseModel(metaclass=abc.ABCMeta):
    pipline: Any = None
    device: Any = None

    @abc.abstractmethod
    def load_model(self, model_path: str, device: str):
        pass

    @abc.abstractmethod
    def generate(self, **kwargs):
        pass

