import time
from typing import Any
from core.models.sequences.llm.base_model import BaseModel
from core.models.backends.loader import BackendLoader

class LlamaModel(BaseModel):
    def load_model(self, model_path: str, device: str, backend: str = "transformer", **kwargs):
        if backend == "transformer" or backend == "pytorch":
            # Default Llama torch params
            if "dtype" not in kwargs:
                import torch
                kwargs["dtype"] = torch.bfloat16
            kwargs.setdefault("trust_remote_code", True)
            kwargs.setdefault("low_cpu_mem_usage", True)
            if device == "cpu":
                 kwargs.setdefault("device_map", "cpu")
            elif device != "cpu" and "device_map" not in kwargs:
                 kwargs.setdefault("device_map", device.lower())

        elif backend == "llama_cpp":
            # kwargs passed to Llama(...)
            pass
            
        elif backend == "openvino":
            pass
            
        elif backend == "onnxruntime" or backend == "onnx":
            if "dtype" not in kwargs:
                import torch
                kwargs["dtype"] = torch.bfloat16
            kwargs.setdefault("trust_remote_code", True)
            
        elif backend == "ipex_llm" or backend == "ipex":
            pass
            
        # Use Loader to get backend instance (lazy import happens inside)
        self.backend = BackendLoader.get_backend(backend, device)

        self.backend.load_text_generation(model_path, **kwargs)
        self.model = self.backend.model
        if hasattr(self.backend, "tokenizer"):
            self.tokenizer = self.backend.tokenizer

    def generate_prompt(self, instruction: str):
        return f"""
                请用中文回答以下问题：
                {instruction}
                """

    def chat(self, history, max_tokens, temperature, top_p, slider_context_times):
        if slider_context_times < 1:
            messages = history[-1:]
        else:
            messages = history[-slider_context_times:]

        generated_tokens_count = 0
        start_time = time.time()
        bot_message = ''
        cost_time, words_count, single_word_cost_time, per_second_tokens = 0, 0, 0, 0
        
        stream_gen = self.backend.stream_chat(
            messages, 
            max_tokens=max_tokens, 
            temperature=temperature, 
            top_p=top_p
        )

        for new_text in stream_gen:
            if not new_text:
                continue

            generated_tokens_count += 1
            
            # Filter special tokens
            if new_text == '<|eot_id|>' or new_text == '<|end_of_text|>':
                break
                
            bot_message += new_text
            
            if "<|eot_id|>" in bot_message or "<|end_of_text|>" in bot_message:
                bot_message = bot_message.replace('<|eot_id|>', '').replace('<|end_of_text|>', '')
                break

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
