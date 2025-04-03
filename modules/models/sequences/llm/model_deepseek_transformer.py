import time
from threading import Thread
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TextIteratorStreamer
from modules.models.sequences.llm.base_model import BaseModel


class DeepSeekTransformerModel(BaseModel):
    def load_model(self, model_path: str, device: str):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model.generation_config = GenerationConfig.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

        model.eval()
        streamer = TextIteratorStreamer(tokenizer,
                                        skip_prompt=True)

        self.model = model
        self.tokenizer = tokenizer
        self.streamer = streamer

    def generate_prompt(self, instruction: str):
        return f"""
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
            return_tensors="pt"
        )

        generate_input = {
            "input_ids": input_ids,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "pad_token_id":  self.tokenizer.eos_token_id,
            "eos_token_id":  self.tokenizer.eos_token_id,
            "streamer": self.streamer,

        }

        thread = Thread(target=self.model.generate, kwargs=generate_input)
        thread.start()

        generated_tokens = []
        start_time = time.time()
        bot_message = ''
        cost_time, words_count, single_word_cost_time, per_second_tokens = 0, 0, 0, 0
        print('[Transformer] Human:', history[-1]["content"])
        print('[Transformer] Assistant: ', end='', flush=True)
        for new_text in self.streamer:
            print(new_text, end='', flush=True)
            if len(new_text) == 0:
                continue

            # 计算生成token 速率
            token_ids = self.tokenizer.encode(new_text, add_special_tokens=False)
            generated_tokens.extend(token_ids)

            if "<think>" in new_text:
                new_text = "<span style='color: blue'>【深度思考】：</span> <br> <blockquote>"
            if "</think>" in new_text:
                new_text = "</blockquote> <span style='color: green'>【推理结果】：</span> <br>"

            if new_text != '<｜end▁of▁sentence｜>':
                bot_message += new_text

            if "<｜end▁of▁sentence｜>" in bot_message:
                bot_message = bot_message.replace('<｜end▁of▁sentence｜>', '')
                end_time = time.time()
                cost_time = round(end_time-start_time, 3)
                trim_message = bot_message.replace("<span style='color: blue'>【深度思考】：</span> <br> <blockquote>", "")
                trim_message = trim_message.replace("</blockquote> <span style='color: green'>【推理结果】：</span> <br>", "").strip()

                words_count = len(trim_message)
                single_word_cost_time = round((end_time-start_time)/len(trim_message), 3)
                per_second_tokens = round(len(generated_tokens) / (end_time-start_time), 3)

            yield bot_message, cost_time, words_count, single_word_cost_time, per_second_tokens

    def release(self):
        del self.model
        del self.streamer
        del self.tokenizer



