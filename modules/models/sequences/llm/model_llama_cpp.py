import time
from modules.models.sequences.llm.base_model import BaseModel
from llama_cpp import Llama


class LlamaCppModel(BaseModel):
    def load_model(self, model_path: str, device: str):
        model = Llama(model_path=model_path)
        self.model = model

    def generate_prompt(self, instruction: str):
        return f"""
                请用中文回答以下问题：
                {instruction}
                """

    def chat(self, history, max_tokens, temperature, top_p, slider_context_times):
        if slider_context_times < 1:
            history = history[-1:]

        response = self.model.create_chat_completion(
            history,
            max_tokens=max_tokens,
            top_k=50,
            top_p=top_p,
            temperature=temperature,
            stream=True
        )

        generated_tokens = []
        start_time = time.time()
        history.append({"role": "assistant", "content": ""})
        cost_time, words_count, single_word_cost_time, per_second_tokens = 0, 0, 0, 0
        print('[LLamaCPP] Human:', history[-1]["content"])
        print('[LLamaCPP] Assistant: ', end='', flush=True)
        for chunk in response:
            if "content" in chunk["choices"][0]["delta"]:
                new_text = chunk["choices"][0]["delta"]["content"]
                print(new_text, end='', flush=True)

                # 计算生成token 速率
                token_ids = self.tokenizer.encode(new_text, add_special_tokens=False)
                generated_tokens.extend(token_ids)

                history[-1]["content"] += new_text
            if "finish_reason" in chunk["choices"][0] and chunk["choices"][0]["finish_reason"] == "stop":
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
