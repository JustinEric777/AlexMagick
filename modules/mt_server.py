import gc
import json
import time
import importlib
from modules.base_server import BaseServer
from config.mt_config import MODEL_LIST
from typing import List

TASK_TYPE = "sequence-mt"


def get_model_list() -> List[str]:
    return [key for key in MODEL_LIST]


class MtServer(BaseServer):
    def load_model(self, task_type: str, model_name: str):
        if self.pipeline_object is not None:
            del self.pipeline_object
            gc.collect()

        if task_type != TASK_TYPE:
            return

        if task_type == TASK_TYPE and model_name not in get_model_list():
            model_name = get_model_list()[0]
        if len(model_name) == 0:
            model_name = get_model_list()[0]

        model_path = MODEL_LIST[model_name]["model_path"]
        model_provider_path = f'{__package__}.{MODEL_LIST[model_name]["model_provider_path"]}'
        model_provider_name = MODEL_LIST[model_name]["model_provider_name"]

        mt_class_name = getattr(importlib.import_module(model_provider_path), model_provider_name)
        self.pipeline_object = mt_class_name()
        self.pipeline_object.load_model(model_path)

        return model_name

    def generate(self, text: str, model_name: str) -> (str, str):
        start_time = time.time()
        outputs = self.pipeline_object.translate(text)
        metric = {
            "model_name": model_name,
            "cost_time": round(time.time()-start_time, 3),
            "words_count": len(outputs),
            "single_word_cost_time":  round((time.time()-start_time)/len(outputs), 3)
        }
        print(f"outputs = {outputs}, metric = {json.dumps(metric)}")

        return outputs, self.get_metric(metric)

    def get_metric(self, metric_info: dict) -> str:
        return f"""
                <span style="color: red">model_name：{metric_info["model_name"]}
                cost_time：{metric_info["cost_time"]} 
                words_count：{metric_info["words_count"]} 
                single_word_cost_time：{metric_info["single_word_cost_time"]}</span>
                """



