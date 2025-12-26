import abc
import os
import json
import time
import gc
import sys
import importlib
from typing import Dict, Any, Optional
from core.models.metrics import Metric, get_format_metric


pipeline_object = None


class BaseServer(abc.ABC):
    task_type: str = ""
    infer_arch: str = ""
    infer_device: str = ""
    model_name: str = ""
    model_version_name: str = ""
    pipeline: Any = None
    model_list: Dict[str, Any] = None

    def __init__(self):
        pass

    def init_model(self, params: dict):
        if "task_type" not in params:
            raise Exception("param task_type is empty")
        if "infer_arch" not in params:
            raise Exception("param infer_arch is empty")
        if "model_name" not in params:
            raise Exception("param model_name is empty")
        if "model_version" not in params:
            raise Exception("param model_version is empty")
        # Subclasses should implement logic to load model using ModelEngine

    @abc.abstractmethod
    def reload_model(self, infer_arch: Optional[str] = None, device: Optional[str] = None,
                     model_name: Optional[str] = None, model_version: Optional[str] = None, default: bool = False):
        pass

    def get_infer_arch_list(self):
        return [key for key in self.model_list]

    def get_arch_model_list(self, infer_arch: Optional[str] = None):
        if infer_arch not in self.model_list:
            return []

        return [key for key in self.model_list[infer_arch]]

    def get_arch_device_list(self, infer_arch: Optional[str] = None):
        if infer_arch not in self.model_list:
            return []

        if infer_arch == "OpenVino":
            return ["CPU", "GPU", "NPU", "AUTO"]

        return ["CPU", "cuda:0", "AUTO"]

    def get_model_list(self, infer_arch: Optional[str] = None, model_name: Optional[str] = None):
        if infer_arch not in self.model_list:
            return []
        if model_name not in self.model_list[infer_arch]:
            return []

        return [name for name in self.model_list[infer_arch][model_name]["model_list"]]

    @abc.abstractmethod
    def generate(self, **kwargs):
        pass

    @Metric()
    def get_metric(self, **kwargs):
        pass

