import json
import time
from typing import List, Tuple
from config.asr_config import MODEL_LIST
from servers.base_server import BaseServer
from core.models.engine import ModelEngine
from core.models.adapters import resolve_model_config
from core.task_monitor import record_task

TASK_TYPE = "audio-asr"


def get_model_list() -> List[str]:
    return [key for key in MODEL_LIST]


class ASRServer(BaseServer):
    def __init__(self):
        super().__init__()
        self.task_type = TASK_TYPE
        self.model_list = MODEL_LIST
        self.engine = None

    def init_model(self, params: dict):
        arch = params.get("infer_arch", "")
        device = params.get("device", "")
        model_name = params.get("model_name", "")
        version = params.get("model_version", "")

        if params.get("task_type") != self.task_type or arch not in self.model_list or model_name not in self.model_list[arch]:
            arch_list = self.get_infer_arch_list()
            arch = arch_list[0] if len(arch_list) > 0 else ""
            
            device_list = self.get_arch_device_list(infer_arch=arch)
            device = device_list[0] if len(device_list) > 0 else ""
            
            model_name_list = self.get_arch_model_list(infer_arch=arch)
            model_name = model_name_list[0] if len(model_name_list) > 0 else ""
            
            version_list = self.get_model_list(infer_arch=arch, model_name=model_name)
            version = version_list[0] if len(version_list) > 0 else ""

        if not arch or arch not in self.model_list:
             print("Warning: No valid model configuration found for ASR. Skipping initialization.")
             return

        model_path = self.model_list[arch][model_name]["model_path"]
        backend, impl = resolve_model_config("asr", arch, model_name)
        self.engine = ModelEngine(
            model_type="asr",
            model_name_or_path=f"{model_path}/{version}",
            device=device,
            backend=backend,
            impl=impl,
        )
        self.pipeline = self.engine.model
        self.infer_arch = arch
        self.device = device
        self.model_name = model_name
        self.model_version_name = version

    def reload_model(self, infer_arch: str = "", device: str = "", model_name: str = "", model_version: str = "", default: bool = False):
        if default:
            arch_list = self.get_infer_arch_list()
            infer_arch = arch_list[0] if len(arch_list) > 0 else ""
            device_list = self.get_arch_device_list(infer_arch)
            device = device_list[0] if len(device_list) > 0 else ""
            model_name_list = self.get_arch_model_list(infer_arch)
            model_name = model_name_list[0] if len(model_name_list) > 0 else ""
            model_version_list = self.get_model_list(infer_arch, model_name)
            model_version = model_version_list[0] if len(model_version_list) > 0 else ""
        
        if not infer_arch or infer_arch not in self.model_list:
             print("Warning: No valid model configuration found for ASR. Skipping reload.")
             return infer_arch, device, model_name, model_version

        model_path = self.model_list[infer_arch][model_name]["model_path"]
        backend, impl = resolve_model_config("asr", infer_arch, model_name)
        self.engine = ModelEngine(
            model_type="asr",
            model_name_or_path=f"{model_path}/{model_version}",
            device=device,
            backend=backend,
            impl=impl,
        )
        self.pipeline = self.engine.model
        self.infer_arch = infer_arch
        self.device = device
        self.model_name = model_name
        self.model_version_name = model_version
        return infer_arch, device, model_name, model_version

    @record_task("ASR", "Speech Recognition")
    def generate(self, audio: str, model_name: str):
        start_time = time.time()
        outputs = self.pipeline.generate(audio)
        metric = {
            "model_name": model_name,
            "cost_time": round(time.time() - start_time, 3),
            "words_count": len(outputs),
            "single_word_cost_time": round((time.time() - start_time) / max(len(outputs), 1), 3),
        }
        print(f"outputs = {outputs}, metric = {json.dumps(metric)}")
        return outputs, self.get_metric(metric)

    def get_metric(self, metric_info: dict):
        return f"""
                <span style="color: red">model_name：{metric_info["model_name"]}
                cost_time：{metric_info["cost_time"]} 
                words_count：{metric_info["words_count"]} 
                single_word_cost_time：{metric_info["single_word_cost_time"]}</span>
                """

