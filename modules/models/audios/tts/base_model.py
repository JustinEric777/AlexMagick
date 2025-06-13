import abc
import os
import sys
from typing import Any
sub_module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
if sub_module_path not in sys.path:
    sys.path.insert(0, sub_module_path)

AUDIO_PATH = "storage/audio/tts"


class BaseModel(metaclass=abc.ABCMeta):
    model: Any = None
    tokenizer: Any = None
    processor: Any = None
    vocoder: Any = None
    device: Any = None
    dtype: Any = None

    @abc.abstractmethod
    def load_model(self, model_path: str, device: str):
        pass

    @abc.abstractmethod
    def inference(self, text: [str], sample_wav: str):
        pass

