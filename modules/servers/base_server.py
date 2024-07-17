import abc
from typing import Any


class BaseServer(abc.ABC):
    pipeline_object: Any = None

    @abc.abstractmethod
    def load_model(self, **kwargs):
        pass

    @abc.abstractmethod
    def generate(self, **kwargs):
        pass

    @abc.abstractmethod
    def get_metric(self, **kwargs):
        pass

