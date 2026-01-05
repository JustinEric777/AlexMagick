from config.multimodal_llm_config import MODEL_LIST, TASK_TYPE
from servers.base_server import BaseServer
from core.monitor import record_task


def remove_metric(history):
    messages = [one_message.copy() for one_message in history]
    for line in messages:
        if line["role"] == "assistant" and type(line["content"]) is str and line["content"] is not None:
            arr = line["content"].split("\n")
            line["content"] = arr[0]
    return messages


class MultimodalLLMServer(BaseServer):
    TASK_TYPE = TASK_TYPE
    MODEL_LIST = MODEL_LIST

    def __init__(self):
        super().__init__()

    @record_task("MultimodalLLM", "Multimodal Generation")
    def generate(self, history, max_tokens, temperature, top_p, slider_context_times, return_audio):
        if not self.pipeline:
             yield history
             return

        messages = remove_metric(history)
        yield_history = [m.copy() for m in messages]
        yield_history.append({"role": "assistant", "content": ""})
        
        for message, cost_time, words_count, single_word_cost_time, output_tokens_count, per_second_tokens in self.pipeline.chat(
                messages,
                max_tokens,
                temperature,
                top_p,
                slider_context_times,
                return_audio
        ):
            if isinstance(message, dict) and "type" in message and message["type"] == "audio":
                yield_history.append({"role": "assistant", "content": {"path": message["data"]}})
            else:
                yield_history[-1]["content"] = message
                if cost_time != 0:
                    metrics = {
                        "cost_time": f"{cost_time}s",
                        "words_count": words_count,
                        "output_tokens": output_tokens_count,
                        "tokens_per_sec": f"{per_second_tokens}/s",
                        "model_name": self.model_name,
                        "model_version": self.model_version_name,
                        "device": self.device
                    }
                    yield_history[-1]["content"] += self.format_metric(metrics)

            yield yield_history
