from core.models.sequences.llm.base_model import BaseModel
from core.models.backends.loader import BackendLoader

class GPT2Model(BaseModel):
    def load_model(self, model_path: str, device: str, backend: str = "transformer", **kwargs):
        if backend == "transformer" or backend == "pytorch":
            if "dtype" not in kwargs:
                import torch
                kwargs["dtype"] = torch.float32 # GPT2 is often fp32
            kwargs.setdefault("trust_remote_code", True)
            if device != "cpu":
                kwargs.setdefault("device_map", device.lower())
        
        self.backend = BackendLoader.load(backend, device, model_path, **kwargs)
        self.model = self.backend.model
        self.tokenizer = self.backend.tokenizer

    def generate_prompt(self, instruction: str):
        return f"{instruction}"

    def chat(self, history, max_tokens, temperature, top_p, slider_context_times):
        # GPT2 is a base model, not chat, but we adapt
        if slider_context_times < 1:
            messages = history[-1:]
        else:
            messages = history[-slider_context_times:]
            
        # Basic prompt construction for non-chat model
        prompt = "\n".join([m["content"] for m in messages])
        
        # Use backend stream
        stream_gen = self.backend.stream_chat(
            [{"role": "user", "content": prompt}], # Adapter will handle this if needed, or we pass string
            max_tokens=max_tokens, 
            temperature=temperature, 
            top_p=top_p
        )
        
        # ... standard streaming loop ...
        # For simplicity reusing Qwen/Llama loop logic
        import time
        generated_tokens_count = 0
        start_time = time.time()
        bot_message = ''
        
        for new_text in stream_gen:
            if not new_text: continue
            bot_message += new_text
            
            end_time = time.time()
            cost_time = round(end_time - start_time, 3)
            words_count = len(bot_message)
            single_word_cost_time = round((end_time - start_time) / max(1, len(bot_message)), 3)
            per_second_tokens = round(generated_tokens_count / max(0.001, (end_time - start_time)), 3)

            yield bot_message, cost_time, words_count, single_word_cost_time, per_second_tokens

    def release(self):
        if hasattr(self, 'backend'):
            del self.backend
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
