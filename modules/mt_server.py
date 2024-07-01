import gc
import importlib
from config.mt_config import MODEL_LIST
from typing import List

mt_object = None


def get_model_list() -> List[str]:
    return [key for key in MODEL_LIST]


def init_model(params: dict):
    global mt_object

    model_name = params["model_name"]
    if params['task_type'] == "mt" and model_name not in get_model_list():
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


def translate(text: str) -> str:
    global mt_object

    outputs = mt_object.translate(text)
    print("outputs =", outputs)

    return outputs