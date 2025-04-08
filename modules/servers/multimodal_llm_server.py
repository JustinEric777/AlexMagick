from config.multimodal_llm_config import MODEL_LIST, TASK_TYPE
from modules.servers.base_server import BaseServer


def remove_metric(history):
    messages = [one_message.copy() for one_message in history]
    for line in messages:
        if line["role"] == "assistant" and type(line["content"]) is str and line["content"] is not None:
            arr = line["content"].split("\n")
            line["content"] = arr[0]

    return messages


class MultimodalLLMServer(BaseServer):
    def __init__(self):
        super().__init__()
        self.model_list = MODEL_LIST
        self.task_type = TASK_TYPE

    def generate(self, history, max_tokens, temperature, top_p, slider_context_times, return_audio):
        messages = remove_metric(history)
        history.append({"role": "assistant", "content": ""})
        for message, cost_time, words_count, single_word_cost_time, output_tokens_count, per_second_tokens in self.pipeline.chat(
                messages,
                max_tokens,
                temperature,
                top_p,
                slider_context_times,
                return_audio
        ):
            if "type" in message and message["type"] == "audio":
                history.append({"role": "assistant", "content": {"path": message["data"]}})
            else:
                history[-1]["content"] = message
                if cost_time != 0 and words_count != 0 and single_word_cost_time != 0:
                    history[-1]["content"] += self.get_metric(self.infer_arch, self.device, self.model_name,
                                                              self.model_version_name, cost_time,
                                                              words_count, single_word_cost_time, output_tokens_count, per_second_tokens)

            yield history

    def get_metric(self, infer_arch: str, device: str, model_name: str, model_version: str, cost_time: float,
                   words_count: int,
                   single_word_cost_time: float,
                   output_tokens_count: int,
                   per_second_tokens: float, ) -> str:
        return f"""
                <span style="color: red; display:block; float:right; margin-right: 10px">
                infer_arch：{infer_arch}
                device：{device}
                model_name：{model_name}
                model_version：{model_version}
                cost_time：{cost_time} 
                words_count：{words_count} 
                per_second_tokens：{per_second_tokens} tokens / s
                output_tokens_count：{output_tokens_count} 
                single_word_cost_time：{single_word_cost_time}</span>
                """
