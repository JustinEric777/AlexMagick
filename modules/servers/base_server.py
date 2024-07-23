import abc
import os
import gc
import importlib
from typing import Dict, Any, Optional


class BaseServer(abc.ABC):
    task_type: str = ""
    infer_arch: str = ""
    model_name: str = ""
    model_version_name: str = ""
    pipeline_object: Any = None
    model_list: Dict[str, Any] = None

    def __init__(self):
        pass

    def __load_model(self, task_type: Optional[str] = None, infer_arch: Optional[str] = None, model_name: Optional[str] = None, model_version: Optional[str] = None):
        if task_type != self.task_type:
            return

        if self.pipeline_object is not None:
            del self.pipeline_object
            gc.collect()

        model_path = os.path.join(self.model_list[infer_arch][model_name]["model_path"], model_version)
        model_provider_path = self.model_list[infer_arch][model_name]["model_provider_path"]
        model_provider_name = self.model_list[infer_arch][model_name]["model_provider_name"]

        llm_class_name = getattr(importlib.import_module(model_provider_path), model_provider_name)
        pipeline_object = llm_class_name()
        pipeline_object.load_model(model_path)

        self.pipeline_object = pipeline_object
        self.infer_arch = infer_arch
        self.model_name = model_name
        self.model_version_name = model_version

        return infer_arch, model_name, model_version

    def init_model(self, params: dict):
        if len(params) < 4:
            raise Exception("params nums is less ...")
        if "task_type" not in params:
            raise Exception("param task_type is empty")
        if "infer_arch" not in params:
            raise Exception("param infer_arch is empty")
        if "model_name" not in params:
            raise Exception("param model_name is empty")
        if "model_version" not in params:
            raise Exception("param model_version is empty")

        self.__load_model(params["task_type"], params["infer_arch"], params["model_name"], params["model_version"])

    def reload_model(self, infer_arch: Optional[str] = None, model_name: Optional[str] = None, model_version: Optional[str] = None, default: bool = False):
        if default:
            arch_list = self.get_infer_arch_list()
            infer_arch = arch_list[0] if len(arch_list) > 0 else ""

            model_name_list = self.get_arch_model_list(infer_arch)
            model_name = model_name_list[0] if len(model_name_list) > 0 else ""

            model_version_list = self.get_model_list(infer_arch, model_name)
            model_version = model_version_list[0] if len(model_version_list) > 0 else ""

        return self.__load_model(self.task_type, infer_arch, model_name, model_version)

    def get_infer_arch_list(self):
        return [key for key in self.model_list]

    def get_arch_model_list(self, infer_arch: Optional[str] = None):
        if infer_arch not in self.model_list:
            return []

        return [key for key in self.model_list[infer_arch]]

    def get_model_list(self, infer_arch: Optional[str] = None, model_name: Optional[str] = None):
        if infer_arch not in self.model_list:
            return []
        if model_name not in self.model_list[infer_arch]:
            return []

        return [key for key in self.model_list[infer_arch][model_name]["model_list"]]

    @abc.abstractmethod
    def generate(self, **kwargs):
        pass

    @abc.abstractmethod
    def get_metric(self, **kwargs):
        pass



