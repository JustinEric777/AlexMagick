import json
import time
from config.text2img_config import MODEL_LIST, TASK_TYPE
from modules.servers.base_server import BaseServer


class Text2ImageServer(BaseServer):
    def __init__(self):
        super().__init__()
        self.task_type = TASK_TYPE
        self.model_list = MODEL_LIST

    def generate(self, positive_prompt, negative_prompt, seed, guidance_scale, num_inference_steps, width, height, model_name):
        start_time = time.time()
        outputs = self.pipeline_object.generate(positive_prompt, negative_prompt, seed, guidance_scale, num_inference_steps, width, height)
        metric = {
            "model_name": model_name,
            "cost_time": round(time.time()-start_time, 3),
        }
        print(f"outputs = {outputs}, metric = {json.dumps(metric)}")

        return outputs, self.get_metric(metric)

    def get_metric(self, metric_info: dict):
        return f"""
                <span style="color: red">model_name：{metric_info["model_name"]}
                cost_time：{metric_info["cost_time"]}</span>"""
