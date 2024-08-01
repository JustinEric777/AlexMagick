import time
from modules.models.sequences.llm.base_model import BaseModel
from llama_cpp import Llama


class LlamaCppModel(BaseModel):
    def load_model(self, model_path: str):
        model = Llama(model_path=model_path)
        self.model = model

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

        response = self.model.create_chat_completion(
            messages,
            max_tokens=max_tokens,
            top_k=50,
            top_p=top_p,
            temperature=temperature,
            stream=True
        )

        start_time = time.time()
        bot_message = ''
        cost_time, words_count, single_word_cost_time = 0, 0, 0
        print('[LLamaCPP] Human:', history[-1][0])
        print('[LLamaCPP] Assistant: ', end='', flush=True)
        for chunk in response:
            if "content" in chunk["choices"][0]["delta"]:
                new_text = chunk["choices"][0]["delta"]["content"]
                print(new_text, end='', flush=True)
                bot_message += new_text
            if "finish_reason" in chunk["choices"][0] and chunk["choices"][0]["finish_reason"] == "stop":
                end_time = time.time()
                cost_time = round(end_time-start_time, 3)
                words_count = len(bot_message)
                single_word_cost_time = round((end_time-start_time)/len(bot_message), 3)

            yield bot_message, cost_time, words_count, single_word_cost_time


