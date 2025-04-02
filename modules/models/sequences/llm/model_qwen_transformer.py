import time
import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from modules.models.sequences.llm.base_model import BaseModel


class QwenTransformerModel(BaseModel):
    def load_model(self, model_path: str, device: str):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cpu",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        self.model = model
        self.tokenizer = tokenizer
        self.streamer = streamer

    def generate_prompt(self, instruction: str):
        return f"""
                {instruction}
                """

    def chat(self, history, max_tokens, temperature, top_p, slider_context_times):
        if slider_context_times < 1:
            history = history[-1:]

        text = self.tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        input_ids = self.tokenizer(
            [text],
            return_tensors="pt"
        ).to(self.model.device)

        generate_input = {
            **input_ids,
            "max_new_tokens": max_tokens,
            # "top_k": 50,
            # "top_p": top_p,
            "streamer": self.streamer,
            # "temperature": temperature,
        }

        thread = Thread(target=self.model.generate, kwargs=generate_input)
        thread.start()

        generated_tokens = []
        start_time = time.time()
        history.append({"role": "assistant", "content": ""})
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

            history[-1]["content"] += new_text

            end_time = time.time()
            cost_time = round(end_time-start_time, 3)
            words_count = len(history[-1]["content"])
            single_word_cost_time = round((end_time-start_time)/len(history[-1]["content"]), 3)
            per_second_tokens = round(len(generated_tokens) / (end_time-start_time), 3)

            yield history, cost_time, words_count, single_word_cost_time, per_second_tokens

    def release(self):
        del self.model
        del self.streamer
        del self.tokenizer




