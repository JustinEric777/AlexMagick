import time
import torch
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

from modules.models.sequences.llm.base_model import BaseModel


class Glm4vTransformerModel(BaseModel):
    def load_model(self, model_path: str, device: str):
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     device_map="auto",
                                                     torch_dtype=torch.bfloat16,
                                                     low_cpu_mem_usage=True,
                                                     trust_remote_code=True)
        model.eval()
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
        self.model = model
        self.tokenizer = tokenizer
        self.streamer = streamer

    def generate_prompt(self, instruction: str):
        return f"""
                请用中文回答以下问题：
                {instruction}
                """

    def chat(self, history, max_tokens, temperature, top_p, slider_context_times):
        messages = []

        history_true = history[1:-1]
        if slider_context_times > 0:
            for one_chat in history_true[-slider_context_times:]:
                one_message_user = {"role": "user", "content": one_chat[0].replace('<br>', '\n')}
                messages.append(one_message_user)
                one_message_system = {"role": "assistant", "content": one_chat[1].replace('<br>', '\n')}
                messages.append(one_message_system)

        input_message = {"role": "user", "content": self.generate_prompt(history[-1][0].replace('<br>', '\n'))}
        messages.append(input_message)

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
        print('Human:', history[-1][0])
        print('Assistant: ', end='', flush=True)
        for new_text in self.streamer:
            print(new_text, end='', flush=True)
            if len(new_text) == 0:
                continue
            # 计算生成token 速率
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
        del self.tokenizer
        del self.streamer

