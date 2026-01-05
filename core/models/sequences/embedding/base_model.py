import abc
from typing import Any
from core.models.loader import BaseModel as LoaderBaseModel


class BaseModel(LoaderBaseModel):
    pooling_method: Any = None

    @abc.abstractmethod
    def encode(self, text: str):
        pass
