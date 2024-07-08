import abc
from typing import Any


class BaseServer(abc.ABC):
    pipeline_object: Any = None
    task_type: Any = None

    def load_model(self, **kwargs):
        pass

    def generate(self, **kwargs):
        pass

    def get_metric(self, **kwargs):
        pass

