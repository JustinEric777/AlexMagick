import time
from core.models.multimodals.llm.base_model import BaseModel
from core.models.backends.loader import BackendLoader


def format_history(history: list):
    messages = []
    for item in history:
        if isinstance(item.get("content"), str) and len(item["content"].strip()) > 0:
            messages.append({"role": item.get("role", "user"), "content": item["content"]})
    return messages


class QwenOmiLLamaCppModel(BaseModel):
    def load_model(self, model_path: str, device: str, backend: str = "llama_cpp", **kwargs):
        # Explicitly use llama_cpp backend, but via loader
        self.backend = BackendLoader.load(backend, device, model_path, **kwargs)
        self.model = self.backend.model

    def generate_prompt(self, instruction: str):
        return f"""
                {instruction}
                """

    def chat(self, history, max_tokens, temperature, top_p, slider_context_times, return_audio=False):
        messages = [
            {"role": "system", "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group."},
        ]
        messages += format_history(history)
        prompt = "\n".join(
            [f"{m['role']}: {m['content']}" for m in messages] + ["assistant:"]
        )

        start_time = time.time()
        # Use backend to create completion
        out = self.model.create_completion(
            prompt=prompt,
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=int(max_tokens),
        )
        text = out.get("choices", [{}])[0].get("text", "").strip()
        end_time = time.time()
        cost_time = round(end_time - start_time, 3)
        words_count = len(text)
        single_word_cost_time = 0 if words_count == 0 else round((end_time - start_time) / words_count, 3)
        output_token_ids_count = len(out.get("tokens", [])) if "tokens" in out else 0
        per_second_tokens = 0 if cost_time == 0 else round(output_token_ids_count / cost_time, 3)
        yield text, cost_time, words_count, single_word_cost_time, output_token_ids_count, per_second_tokens

    def release(self):
        del self.model
        if hasattr(self, 'backend'):
            del self.backend
