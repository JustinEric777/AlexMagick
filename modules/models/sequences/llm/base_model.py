import abc
from typing import Any


class BaseModel(metaclass=abc.ABCMeta):
    model: Any = None
    tokenizer: Any = None
    streamer: Any = None

    @abc.abstractmethod
    def load_model(self, model_path: str):
        pass

    @abc.abstractmethod
    def generate_prompt(self, instruction: str):
        pass

    @abc.abstractmethod
    def chat(self, history, max_tokens, temperature, top_p, slider_context_times):
        pass

