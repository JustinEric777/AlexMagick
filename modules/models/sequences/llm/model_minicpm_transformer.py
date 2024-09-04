import time
from threading import Thread
import torch
from transformers import AutoModel, AutoTokenizer, TextIteratorStreamer
from modules.models.sequences.llm.base_model import BaseModel


class MiniCPMTransformerModel(BaseModel):
    def load_model(self, model_path: str):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            attn_implementation='sdpa',
            torch_dtype=torch.bfloat16
        )
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
        messages = []
        history_true = history[1:-1]
        if slider_context_times > 0:
            for one_chat in history_true[-slider_context_times:]:
                one_message_user = {"role": "user", "content": [one_chat[0].replace('<br>', '\n')]}
                messages.append(one_message_user)
                one_message_system = {"role": "assistant", "content": [one_chat[1].replace('<br>', '\n')]}
                messages.append(one_message_system)

        input_message = {"role": "user", "content": [self.generate_prompt(history[-1][0].replace('<br>', '\n'))]}
        messages.append(input_message)

        generate_input = {
            "tokenizer": self.tokenizer,
            "image": None,
            "msgs": messages,
            "sampling": True,
            "stream": True,
            "top_k": 50,
            "top_p": top_p,
            "max_new_tokens": max_tokens,
            "streamer": self.streamer,
            "temperature": temperature,
        }

        thread = Thread(target=self.model.generate, kwargs=generate_input)
        thread.start()

        # res = self.model.chat(
        #     image=None,
        #     msgs=messages,
        #     tokenizer=self.tokenizer,
        #     sampling=True,
        #     stream=True
        # )

        start_time = time.time()
        bot_message = ''
        cost_time, words_count, single_word_cost_time = 0, 0, 0
        print('[Transformer] Human:', history[-1][0])
        print('[Transformer] Assistant: ', end='', flush=True)
        for new_text in self.streamer:
            print(new_text, end='', flush=True)
            bot_message += new_text

            end_time = time.time()
            cost_time = round(end_time-start_time, 3)
            words_count = len(bot_message)
            single_word_cost_time = round((end_time-start_time)/len(bot_message), 3)

            yield bot_message, cost_time, words_count, single_word_cost_time





