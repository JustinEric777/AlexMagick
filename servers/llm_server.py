from config.llm_config import MODEL_LIST, TASK_TYPE
from servers.base_server import BaseServer
from core.storage.history import HistoryStore
from core.monitor import record_task


class LLMServer(BaseServer):
    TASK_TYPE = TASK_TYPE
    MODEL_LIST = MODEL_LIST

    def __init__(self):
        super().__init__()
        self.history = HistoryStore()

    @record_task("LLM", "LLM Generation")
    def generate(self, history, max_tokens, temperature, top_p, slider_context_times):
        # 1. Prepare history
        messages = []
        for one_message in history:
            if isinstance(one_message, dict):
                 messages.append(one_message.copy())

        for line in messages:
            if line.get("content") is not None:
                arr = line["content"].split("\n")
                line["content"] = arr[0]
        
        yield_history = [h.copy() for h in messages]
        yield_history.append({"role": "assistant", "content": ""})
        
        # 2. Check pipeline
        if not self.pipeline:
             yield yield_history
             return

        # 3. Stream generation
        last_metrics = {}
        # LLM streaming doesn't fit neatly into pre/post_generate because it yields incrementally
        # But we can reuse format_metric from base
        
        for message, cost_time, words_count, single_word_cost_time, per_second_tokens in self.pipeline.chat(
                messages,
                max_tokens,
                temperature,
                top_p,
                slider_context_times
        ):
            yield_history[-1]["content"] = message
            
            if cost_time != 0 and words_count != 0:
                last_metrics = {
                    "cost_time": f"{cost_time}s",
                    "words_count": words_count,
                    "single_word_cost_time": single_word_cost_time,
                    "per_second_tokens": f"{per_second_tokens} tokens/s",
                    "model_name": self.model_name,
                    "model_version": self.model_version_name,
                    "device": self.device
                }
                # Use BaseServer helper
                metric_html = self.format_metric(last_metrics)
                yield_history[-1]["content"] += metric_html
            
            yield yield_history
            
        try:
            self.history.save_llm_record(
                self.infer_arch,
                self.device,
                self.model_name,
                self.model_version_name,
                messages,
                yield_history[-1]["content"],
                last_metrics,
            )
        except Exception:
            pass
