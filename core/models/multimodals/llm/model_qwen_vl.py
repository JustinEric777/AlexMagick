from core.models.multimodals.llm.base_model import BaseModel
from core.models.backends.loader import BackendLoader
import time

class QwenVLModel(BaseModel):
    def load_model(self, model_path: str, device: str, backend: str = "transformer", **kwargs):
        if backend == "transformer" or backend == "pytorch":
            import torch
            kwargs.setdefault("dtype", torch.bfloat16)
            kwargs.setdefault("trust_remote_code", True)
            if device != "cpu":
                kwargs.setdefault("device_map", device.lower())
        
        self.backend = BackendLoader.load(backend, device, model_path, **kwargs)
        self.model = self.backend.model
        self.tokenizer = self.backend.tokenizer

    def generate_prompt(self, instruction: str):
        return f"{instruction}"

    def chat(self, history, max_tokens, temperature, top_p, slider_context_times, return_audio=False):
        # Qwen-VL chat logic
        if slider_context_times < 1:
            messages = history[-1:]
        else:
            messages = history[-slider_context_times:]
            
        # Assuming backend handles Qwen-VL specifics or we use standard chat
        # Qwen-VL might need image inputs, but here we assume text-only or pre-processed inputs for simplicity
        # or that BackendLoader's stream_chat can handle it if inputs are formatted correctly.
        
        stream_gen = self.backend.stream_chat(
            messages, 
            max_tokens=max_tokens, 
            temperature=temperature, 
            top_p=top_p
        )
        
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
            # Placeholder for output_token_ids_count
            output_token_ids_count = generated_tokens_count 
            per_second_tokens = round(generated_tokens_count / max(0.001, (end_time - start_time)), 3)

            yield bot_message, cost_time, words_count, single_word_cost_time, output_token_ids_count, per_second_tokens

    def release(self):
        if hasattr(self, 'backend'):
            del self.backend
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
