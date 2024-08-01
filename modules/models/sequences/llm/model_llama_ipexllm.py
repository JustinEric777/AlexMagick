import torch
import time
from modules.models.sequences.llm.base_model import BaseModel
from ipex_llm import optimize_model
from ipex_llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


class LlamaIpexLLMModel(BaseModel):
    def load_model(self, model_path: str):
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     device_map="auto",
                                                     torch_dtype=torch.bfloat16,
                                                     load_in_low_bit="bf16",
                                                     low_cpu_mem_usage=True,
                                                     trust_remote_code=True)

        model = optimize_model(model)

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = model
        self.tokenizer = tokenizer

    def generate_prompt(self, instruction: str):
        return f"""
                请用中文回答以下问题：
                {instruction}
                """

    def chat(self, history, max_tokens, temperature, top_p, slider_context_times):
        messages = [
            {"role": "system", "content": ""}
        ]

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
            return_tensors="pt"
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        response = self.model.generate(
            input_ids,
            eos_token_id=terminators,
            max_new_tokens=max_tokens,
            do_sample=True,
            top_k=50,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=self.tokenizer.pad_token_id
        )

        start_time = time.time()
        bot_message = ''
        cost_time, words_count, single_word_cost_time = 0, 0, 0
        print('[IpexLLM] Human:', history[-1][0])
        print('[IpexLLM] Assistant: ', end='', flush=True)
        for chunk in response:
            print(chunk)

            end_time = time.time()
            cost_time = round(end_time-start_time, 3)
            words_count = len(bot_message)
            single_word_cost_time = round((end_time-start_time)/len(bot_message), 3)
            bot_message = chunk
            yield bot_message, cost_time, words_count, single_word_cost_time

