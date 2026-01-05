import time
from typing import Any
from core.models.sequences.llm.base_model import BaseModel
from core.models.backends.loader import BackendLoader

class DeepSeekModel(BaseModel):
    def load_model(self, model_path: str, device: str, backend: str = "transformer", **kwargs):
        # Use centralized Loader to init backend and load model
        # The loader now handles default params for each backend (like dtype, trust_remote_code, etc.)
        self.backend = BackendLoader.load(backend, device, model_path, **kwargs)
        
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
        
        # Use backend unified stream interface
        stream_gen = self.backend.stream_chat(
            messages, 
            max_tokens=max_tokens, 
            temperature=temperature, 
            top_p=top_p
        )

        for new_text in stream_gen:
            if not new_text:
                continue

            generated_tokens_count += 1 # Estimation or use tokenizer len if critical

            # DeepSeek specific post-processing
            if "<think>" in new_text:
                new_text = "<span style='color: blue'>【深度思考】：</span> <br> <blockquote>"
            if "</think>" in new_text:
                new_text = "</blockquote> <span style='color: green'>【推理结果】：</span> <br>"
            
            # Filter special tokens if they leak through
            if new_text == '<｜end▁of▁sentence｜>':
                continue
                
            bot_message += new_text

            if "<｜end▁of▁sentence｜>" in bot_message:
                bot_message = bot_message.replace('<｜end▁of▁sentence｜>', '')
                # End of generation logic handled below normally, but here we can break early if token found
                break

            # Metrics calculation
            end_time = time.time()
            cost_time = round(end_time - start_time, 3)
            
            # Clean message for word counting
            trim_message = bot_message.replace("<span style='color: blue'>【深度思考】：</span> <br> <blockquote>", "")
            trim_message = trim_message.replace("</blockquote> <span style='color: green'>【推理结果】：</span> <br>", "").strip()
            
            words_count = len(trim_message)
            single_word_cost_time = round((end_time - start_time) / max(1, len(trim_message)), 3)
            per_second_tokens = round(generated_tokens_count / max(0.001, (end_time - start_time)), 3)

            yield bot_message, cost_time, words_count, single_word_cost_time, per_second_tokens

    def release(self):
        if hasattr(self, 'backend'):
            del self.backend
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
