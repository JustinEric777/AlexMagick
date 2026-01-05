import time
import torch
from threading import Thread
from transformers import TextIteratorStreamer

from core.models.sequences.llm.base_model import BaseModel
from core.models.backends.loader import BackendLoader


class Glm4vTransformerModel(BaseModel):
    def load_model(self, model_path: str, device: str, backend: str = "transformer", **kwargs):
        # GLM4v specific defaults
        if "device_map" not in kwargs:
             kwargs["device_map"] = "auto"
        
        # Use BackendLoader
        self.backend = BackendLoader.load(backend, device, model_path, **kwargs)
        
        self.model = self.backend.model
        self.tokenizer = self.backend.tokenizer
        
        # Re-create streamer as it was used in the original implementation
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)

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

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        ).to(self.model.device)

        generate_input = {
            "input_ids": input_ids,
            "max_length": max_tokens,
            "do_sample": True,
            "streamer": self.streamer,
            "top_k": 1,
            "top_p": top_p,
            "temperature": temperature
        }

        thread = Thread(target=self.model.generate, kwargs=generate_input)
        thread.start()

        generated_tokens = []
        start_time = time.time()
        bot_message = ''
        cost_time, words_count, single_word_cost_time, per_second_tokens = 0, 0, 0, 0
        print('Human:', history[-1]["content"])
        print('Assistant: ', end='', flush=True)
        for new_text in self.streamer:
            print(new_text, end='', flush=True)
            if len(new_text) == 0:
                continue
            token_ids = self.tokenizer.encode(new_text, add_special_tokens=False)
            generated_tokens.extend(token_ids)

            if new_text != '<|eot_id|>':
                bot_message += new_text
            if "<|eot_id|>" in bot_message or "<|end_of_text|>" in bot_message:
                bot_message = bot_message.replace('<|eot_id|>', '')
                bot_message = bot_message.replace('<|end_of_text|>', '')
                end_time = time.time()

                cost_time = round(end_time-start_time, 3)
                words_count = len(bot_message)
                single_word_cost_time = round((end_time-start_time)/len(bot_message), 3)
                per_second_tokens = round(len(generated_tokens) / (end_time-start_time), 3)

            yield bot_message, cost_time, words_count, single_word_cost_time, per_second_tokens

    def release(self):
        del self.model
        del self.streamer
        del self.tokenizer
        if hasattr(self, 'backend'):
            del self.backend
