import abc
from typing import Any
from core.models.loader import BaseModel as LoaderBaseModel


class BaseModel(LoaderBaseModel):
    @abc.abstractmethod
    def text_encode(self, text: str):
        pass

    @abc.abstractmethod
    def video_encode(self, text: str):
        pass
