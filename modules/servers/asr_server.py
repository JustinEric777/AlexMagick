import gc
import json
import time
import importlib
from typing import List
from config.asr_config import MODEL_LIST
from modules.servers.base_server import BaseServer

TASK_TYPE = "audio-asr"


def get_model_list() -> List[str]:
    return [key for key in MODEL_LIST]


class ASRServer(BaseServer):
    def __init__(self):
        super().__init__()
        self.task_type = TASK_TYPE
        self.model_list = MODEL_LIST

    def generate(self, audio: str, model_name: str):
        start_time = time.time()
        outputs = self.pipeline_object.generate(audio)
        metric = {
            "model_name": model_name,
            "cost_time": round(time.time()-start_time, 3),
            "words_count": len(outputs),
            "single_word_cost_time":  round((time.time()-start_time)/len(outputs), 3)
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
