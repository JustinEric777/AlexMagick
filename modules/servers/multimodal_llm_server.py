from config.multimodal_llm_config import MODEL_LIST, TASK_TYPE
from modules.servers.base_server import BaseServer


class MultimodalLLMServer(BaseServer):
    def __init__(self):
        super().__init__()
        self.model_list = MODEL_LIST
        self.task_type = TASK_TYPE

    def generate(self, history, max_tokens, temperature, top_p, slider_context_times):
        messages = [one_message.copy() for one_message in history]
        for line in messages:
            if line[1] is not None:
                arr = line[1].split("\n")
                line[1] = arr[0]
        for message, cost_time, words_count, single_word_cost_time, per_second_tokens in self.pipeline.chat(
                messages,
                max_tokens,
                temperature,
                top_p,
                slider_context_times
        ):
            history[-1][1] = message
            if cost_time != 0 and words_count != 0 and single_word_cost_time != 0:
                history[-1][1] += self.get_metric(self.infer_arch, self.device, self.model_name,
                                                  self.model_version_name, cost_time,
                                                  words_count, single_word_cost_time, per_second_tokens)
            yield history

    def get_metric(self, infer_arch: str, device: str, model_name: str, model_version: str, cost_time: float,
                   words_count: int,
                   single_word_cost_time: float,
                   per_second_tokens: float,) -> str:
        return f"""<span style="color: red; display:block; float:right; margin-right: 10px">
                    infer_arch：{infer_arch}
                    device：{device}
                    model_name：{model_name}
                    model_version：{model_version}
                    cost_time：{cost_time} 
                    words_count：{words_count} 
                    per_second_tokens：{per_second_tokens} tokens / s
                    single_word_cost_time：{single_word_cost_time}</span>
                """