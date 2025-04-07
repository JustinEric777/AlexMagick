import json
import time
from typing import List
from config.tts_config import MODEL_LIST
from modules.servers.base_server import BaseServer

TASK_TYPE = "audio-tts"


def get_model_list() -> List[str]:
    return [key for key in MODEL_LIST]


class TTSServer(BaseServer):
    def __init__(self):
        super().__init__()
        self.task_type = TASK_TYPE
        self.model_list = MODEL_LIST

    def generate(self, texts: [], model_name: str):
        start_time = time.time()
        outputs = self.pipeline.inference(texts)
        metric = {
            "model_name": model_name,
            "text_length": len(texts),
            "cost_time": round(time.time()-start_time, 3),
        }
        print(f"outputs = {outputs}, metric = {json.dumps(metric)}")

        return outputs, self.get_metric(metric)

    def get_metric(self, metric_info: dict):
        return f"""
                <span style="color: red">model_name：{metric_info["model_name"]}
                text_length：{metric_info["text_length"]}
                cost_time：{metric_info["cost_time"]} 
                """
