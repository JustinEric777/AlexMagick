import os
import gc
import importlib
from typing import List
from modules.servers.base_server import BaseServer
from config.llm_config import MODEL_LIST

TASK_TYPE = "sequence-llm"


def get_model_arch_list() -> List[str]:
    return [key for key in MODEL_LIST]


def get_inference_arch_list(arch_model: str) -> List[str]:
    return [key for key in MODEL_LIST[arch_model]]


def get_model_list(arch_model: str, infer_arch: str) -> List[str]:
    return MODEL_LIST[arch_model][infer_arch]["model_list"]


class LLMServer(BaseServer):
    def load_model(self, task_type: str, arch_model: str, infer_arch: str, model_name: str):
        if self.pipeline_object is not None:
            del self.pipeline_object
            gc.collect()

        if task_type != TASK_TYPE:
            return

        if task_type == TASK_TYPE and model_name not in get_model_list(arch_model, infer_arch):
            model_name = get_model_list(arch_model, infer_arch)[0]
        if len(model_name) == 0:
            model_name = get_model_list(arch_model, infer_arch)[0]

        model_path = os.path.join(MODEL_LIST[arch_model][infer_arch]["model_path"], model_name)
        model_provider_path = f'{__package__}.{MODEL_LIST[arch_model][infer_arch]["model_provider_path"]}'
        model_provider_name = MODEL_LIST[arch_model][infer_arch]["model_provider_name"]

        llm_class_name = getattr(importlib.import_module(model_provider_path), model_provider_name)
        self.pipeline_object = llm_class_name()
        self.pipeline_object.load_model(model_path)

        return arch_model, infer_arch, model_name

    def generate(self, history, temperature, top_p, slider_context_times, infer_arch, model_name):
        messages = [one_message.copy() for one_message in history]
        for line in messages:
            if line[1] is not None:
                arr = line[1].split("\n")
                line[1] = arr[0]
        for message, cost_time, words_count, single_word_cost_time in self.pipeline_object.chat(
            messages,
            temperature,
            top_p,
            slider_context_times
        ):
            history[-1][1] = message
            if cost_time != 0 and words_count != 0 and single_word_cost_time != 0:
                history[-1][1] += self.get_metric(infer_arch, model_name, cost_time, words_count, single_word_cost_time)
            yield history

    def get_metric(self, infer_arch: str, model_name: str, cost_time: float, words_count: int, single_word_cost_time: float) -> str:
        return f"""
                <span style="color: red; display:block; float:right; margin-right: 10px">
                infer_arch：{infer_arch}
                model_name：{model_name}
                cost_time：{cost_time} 
                words_count：{words_count} 
                single_word_cost_time：{single_word_cost_time}</span>
                """
