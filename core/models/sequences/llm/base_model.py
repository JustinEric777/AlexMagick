import abc
from typing import Any
from core.models.loader import BaseModel as LoaderBaseModel


class BaseModel(LoaderBaseModel):
    streamer: Any = None

    @abc.abstractmethod
    def generate_prompt(self, instruction: str):
        pass

    @abc.abstractmethod
    def chat(self, history, max_tokens, temperature, top_p, slider_context_times):
        pass
