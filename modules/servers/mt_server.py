import json
import time
from modules.servers.base_server import BaseServer
from config.mt_config import MODEL_LIST, TASK_TYPE


class MtServer(BaseServer):
    def __init__(self):
        super().__init__()
        self.task_type = TASK_TYPE
        self.model_list = MODEL_LIST

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



