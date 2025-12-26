import abc
import os
import sys
from typing import Any, List, Union

sub_module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
if sub_module_path not in sys.path:
    sys.path.insert(0, sub_module_path)


class BaseModel(metaclass=abc.ABCMeta):
    model: Any = None
    processor: Any = None
    device: Any = None
    dtype: Any = None

    @abc.abstractmethod
    def load_model(self, model_path: str, device: str):
        pass

    @abc.abstractmethod
    def get_text_features(self, texts: Union[str, List[str]]):
        pass

    @abc.abstractmethod
    def get_audio_features(self, audios: Union[str, List[str]]):
        pass

    @abc.abstractmethod
    def calculate_sim(self, texts: Union[str, List[str]], audios: Union[str, List[str]]):
        pass

