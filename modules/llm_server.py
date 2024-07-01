import os
import gc
import importlib
from typing import List
from config.llm_config import MODEL_LIST

llm_object = None


def get_model_arch_list() -> List[str]:
    return [key for key in MODEL_LIST]


def get_inference_arch_list(arch_model: str) -> List[str]:
    return [key for key in MODEL_LIST[arch_model]]


def get_model_list(arch_model: str, infer_arch: str) -> List[str]:
    return MODEL_LIST[arch_model][infer_arch]["model_list"]


def get_metric(infer_arch: str, model_name: str, cost_time: float, words_count: int, single_word_cost_time: float) -> str:
    return f"""
                <span style="color: red; display:block; float:right; margin-right: 10px">
                推理框架：{infer_arch}
                模型名称：{model_name}
                生成耗时：{cost_time} 
                文字长度：{words_count} 
                单字耗时：{single_word_cost_time}</span>
                """


def init_model(params: dict):
    global llm_object

    model_name = params["model_name"]
    arch_model = params["arch_model"]
    infer_arch = params["infer_arch"]
    model_path = os.path.join(MODEL_LIST[arch_model][infer_arch]["model_path"], model_name)
    model_provider_path = f'{__package__}.{MODEL_LIST[arch_model][infer_arch]["model_provider_path"]}'
    model_provider_name = MODEL_LIST[arch_model][infer_arch]["model_provider_name"]

    llm_class_name = getattr(importlib.import_module(model_provider_path), model_provider_name)
    llm_object = llm_class_name()
    llm_object.load_model(model_path)


def reload_model(arch_model: str, infer_arch: str, model_name: str):
    global llm_object

    # uninstall last model
    del llm_object
    gc.collect()

    model_path = os.path.join(MODEL_LIST[arch_model][infer_arch]["model_path"], model_name)
    model_provider_path = f'{__package__}.{MODEL_LIST[arch_model][infer_arch]["model_provider_path"]}'
    model_provider_name = MODEL_LIST[arch_model][infer_arch]["model_provider_name"]

    llm_class_name = getattr(importlib.import_module(model_provider_path), model_provider_name)
    llm_object = llm_class_name()
    llm_object.load_model(model_path)

    return arch_model, infer_arch, model_name


def chat(history, temperature, top_p, slider_context_times, infer_arch, model_name):
    global llm_object

    messages = [one_message.copy() for one_message in history]
    for line in messages:
        if line[1] is not None:
            arr = line[1].split("\n")
            line[1] = arr[0]
    for message, cost_time, words_count, single_word_cost_time in llm_object.chat(messages, temperature, top_p,
                                                                                    slider_context_times):
        history[-1][1] = message
        if cost_time != 0 and words_count != 0 and single_word_cost_time != 0:
            history[-1][1] += get_metric(infer_arch, model_name, cost_time, words_count, single_word_cost_time)
        yield history
