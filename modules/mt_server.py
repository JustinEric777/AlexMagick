import gc
import json
import time
import importlib
from config.mt_config import MODEL_LIST
from typing import List

TASK_TYPE = "sequence-mt"
mt_object = None


def get_model_list() -> List[str]:
    return [key for key in MODEL_LIST]


def init_model(params: dict):
    global mt_object

    model_name = params["model_name"]
    print(params)
    if params['task_type'] == TASK_TYPE and model_name not in get_model_list():
        model_name = get_model_list()[0]

    model_path = MODEL_LIST[model_name]["model_path"]
    model_provider_path = f'{__package__}.{MODEL_LIST[model_name]["model_provider_path"]}'
    model_provider_name = MODEL_LIST[model_name]["model_provider_name"]

    mt_class_name = getattr(importlib.import_module(model_provider_path), model_provider_name)
    mt_object = mt_class_name()
    mt_object.load_model(model_path)


def reload_model(model_name: str) -> str:
    global mt_object

    # uninstall last model
    del mt_object
    gc.collect()

    if len(model_name) == 0:
        model_name = get_model_list()[0]

    model_path = MODEL_LIST[model_name]["model_path"]
    model_provider_path = f'{__package__}.{MODEL_LIST[model_name]["model_provider_path"]}'
    model_provider_name = MODEL_LIST[model_name]["model_provider_name"]

    mt_class_name = getattr(importlib.import_module(model_provider_path), model_provider_name)
    mt_object = mt_class_name()
    mt_object.load_model(model_path)

    return model_name


def get_metric(metric_info: dict) -> str:
    return f"""
            <span style="color: red">model_name：{metric_info["model_name"]}
            cost_time：{metric_info["cost_time"]} 
            words_count：{metric_info["words_count"]} 
            single_word_cost_time：{metric_info["single_word_cost_time"]}</span>
            """


def translate(text: str, model_name: str) -> (str, str):
    global mt_object

    start_time = time.time()
    outputs = mt_object.translate(text)
    metric = {
        "model_name": model_name,
        "cost_time": round(time.time()-start_time, 3),
        "words_count": len(outputs),
        "single_word_cost_time":  round((time.time()-start_time)/len(outputs), 3)
    }
    print(f"outputs = {outputs}, metric = {json.dumps(metric)}")

    return outputs, get_metric(metric)
