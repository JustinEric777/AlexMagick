import time
from threading import Thread
from modules.models.sequences.llm.base_model import BaseModel
from transformers import AutoTokenizer, TextIteratorStreamer
from optimum.intel.openvino import OVModelForCausalLM


class LlamaOpenvinoModel(BaseModel):
    def load_model(self, model_path: str, device: str):
        print(model_path)
        model = OVModelForCausalLM.from_pretrained(
            model_path,
            use_cache=True,
            export=False
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        streamer = TextIteratorStreamer(tokenizer,
                                        skip_prompt=True)

        self.model = model
        self.streamer = streamer
        self.tokenizer = tokenizer

    def generate_prompt(self, instruction: str):
        return f"""
                请用中文回答以下问题：
                {instruction}
                """

    def chat(self, history, max_tokens, temperature, top_p, slider_context_times):
        if slider_context_times < 1:
            history = history[-1:]

        input_ids = self.tokenizer.apply_chat_template(
            history,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        generate_input = {
            "input_ids": input_ids,
            "max_new_tokens": max_tokens,
            "do_sample": True,
            "top_k": 50,
            "top_p": top_p,
            "streamer": self.streamer,
            "temperature": temperature,
            "eos_token_id": terminators,
            "pad_token_id":  self.tokenizer.eos_token_id
        }

        thread = Thread(target=self.model.generate, kwargs=generate_input)
        thread.start()

        generated_tokens = []
        start_time = time.time()
        history.append({"role": "assistant", "content": ""})
        cost_time, words_count, single_word_cost_time, per_second_tokens = 0, 0, 0, 0
        print('[OpenVino] Human:', history[-1]["content"])
        print('[OpenVino] Assistant: ', end='', flush=True)
        for new_text in self.streamer:
            print(new_text, end='', flush=True)
            if len(new_text) == 0:
                continue

            # 计算生成token 速率
            token_ids = self.tokenizer.encode(new_text, add_special_tokens=False)
            generated_tokens.extend(token_ids)

            if new_text != '<|eot_id|>':
                history[-1]["content"] += new_text
            if "<|eot_id|>" in history[-1]["content"] or "<|end_of_text|>" in history[-1]["content"]:
                history[-1]["content"] = history[-1]["content"].replace('<|eot_id|>', '')
                history[-1]["content"] = history[-1]["content"].replace('<|end_of_text|>', '')
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
