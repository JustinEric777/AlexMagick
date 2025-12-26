import time
import json
from typing import Any

def get_format_metric(metric_info: dict):
    metric_info_line = ""
    if len(metric_info) == 0:
        return metric_info_line

    for key, val in metric_info.items():
        metric_info_line += f"{key}: {val} \n"

    return f"""<span style="color: red">{metric_info_line}</span>"""


class Metric:
    def __init__(self, **kwargs):
        self.params = kwargs

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            start_time = time.time()

            result = func(*args, **kwargs)

            # args[0] is 'self' (the server instance)
            server_instance = args[0]
            
            metric = {
                "infer_arch": getattr(server_instance, "infer_arch", ""),
                "model_name": getattr(server_instance, "model_name", ""),
                "model_version": getattr(server_instance, "model_version_name", ""),
                "cost_time": round(time.time() - start_time, 3)
            }

            if hasattr(server_instance, "task_type") and "sequence-mt" in server_instance.task_type:
                metric["words_count"] = len(result)
                metric["single_word_cost_time"] = round(metric["cost_time"] / max(metric["words_count"], 1), 3)

            # Avoid print in production code, but keeping it for now as per original
            # print(f"[{metric['infer_arch']}]: inputs = {json.dumps(args[1:])}, outputs = {result}, metric = {json.dumps(metric)}")

            return result, get_format_metric(metric)

        return wrapper
