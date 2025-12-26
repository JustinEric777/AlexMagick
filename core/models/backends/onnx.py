from __future__ import annotations

from typing import Any, Iterator
from threading import Thread

from transformers import AutoTokenizer, TextIteratorStreamer
from optimum.onnxruntime import ORTModelForCausalLM

from .base import BackendRunner


class ONNXRuntimeBackend(BackendRunner):
    def load_text_generation(self, model_name_or_path: str, **kwargs: Any) -> None:
        # Pass additional kwargs to from_pretrained if needed (e.g. provider options)
        self.model = ORTModelForCausalLM.from_pretrained(model_name_or_path, use_cache=True, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def generate_text(self, input_ids, **gen_kwargs: Any):
        return self.model.generate(input_ids=input_ids, **gen_kwargs)

    def stream_chat(self, messages: list[dict], **kwargs) -> Iterator[str]:
        if not hasattr(self, "tokenizer") or not hasattr(self, "model"):
            raise RuntimeError("Model or Tokenizer not initialized.")
            
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        gen_kwargs = {
            "input_ids": input_ids,
            "streamer": streamer,
            "max_new_tokens": kwargs.get("max_tokens", 512),
            "do_sample": True,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        
        for new_text in streamer:
            yield new_text

