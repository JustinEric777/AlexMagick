import abc
from typing import Any


class BaseModel(metaclass=abc.ABCMeta):
    model: Any = None
    tokenizer: Any = None
    processor: Any = None
    device: Any = None
    dtype: Any = None

    @abc.abstractmethod
    def load_model(self, model_path: str, device: str):
        pass

    @abc.abstractmethod
    def inference(self, text: [str], audio_path: str, sample_wav: str):
        pass

