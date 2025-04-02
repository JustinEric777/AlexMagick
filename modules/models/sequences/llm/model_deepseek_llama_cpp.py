import time
from modules.models.sequences.llm.base_model import BaseModel
from llama_cpp import Llama


class DeepSeekLlamaCppModel(BaseModel):
    def load_model(self, model_path: str, device: str):
        model = Llama(model_path=model_path, n_ctx=2048, n_threads=16)
        self.model = model

    def generate_prompt(self, instruction: str):
        return f"""
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
                if len(new_text) == 0:
                    continue

                # 计算生成token 速率
                token_ids = self.model.tokenize(str.encode(new_text))
                generated_tokens.extend(token_ids)

                if "<think>" in new_text:
                    new_text = " <span style='color: blue'>【深度思考】：</span> <br> <blockquote>"
                if "</think>" in new_text:
                    new_text = "</blockquote> <span style='color: green'>【推理结果】：</span> <br>"

                history[-1]["content"] += new_text

            if "finish_reason" in chunk["choices"][0] and chunk["choices"][0]["finish_reason"] in ["stop", "length"]:
                end_time = time.time()
                cost_time = round(end_time-start_time, 3)
                trim_message = history[-1]["content"].replace("<span style='color: blue'>【深度思考】：</span> <br> <blockquote>", "")
                trim_message = trim_message.replace("</blockquote> <span style='color: green'>【推理结果】：</span> <br>", "").strip()
                words_count = len(trim_message)

                single_word_cost_time = round((end_time-start_time)/len(trim_message), 3)
                per_second_tokens = round(len(generated_tokens) / (end_time-start_time), 3)

            yield history, cost_time, words_count, single_word_cost_time, per_second_tokens

    def release(self):
        del self.model
