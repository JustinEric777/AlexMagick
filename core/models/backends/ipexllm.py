from __future__ import annotations

from typing import Any, Iterator
import torch
from transformers import AutoTokenizer

from .base import BackendRunner


class IpexLLMBackend(BackendRunner):
    def load_text_generation(self, model_name_or_path: str, **kwargs: Any) -> None:
        try:
            from ipex_llm import optimize_model
            from ipex_llm.transformers import AutoModelForCausalLM
        except ImportError:
            raise ImportError("ipex-llm is not installed. Please install it to use IpexLLMBackend.")

        # Default params from original implementation
        load_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
            "load_in_low_bit": "bf16",
            "low_cpu_mem_usage": True,
            "trust_remote_code": True
        }
        load_kwargs.update(kwargs)

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
        self.model = optimize_model(self.model)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, 
            trust_remote_code=load_kwargs.get("trust_remote_code", True)
        )

    def generate_text(self, input_ids, **gen_kwargs: Any):
        return self.model.generate(input_ids, **gen_kwargs)

    def stream_chat(self, messages: list[dict], **kwargs) -> Iterator[str]:
        # IPEX-LLM usually works with standard transformers generate but optimization might affect streaming
        # The original implementation used model.generate and yielded chunks? 
        # Wait, the original `model_llama_ipexllm.py` implementation:
        # response = self.model.generate(...) 
        # for chunk in response: ...
        # Standard transformers generate returns a Tensor, not an iterator, unless using a Streamer.
        # The original code seems to assume `response` is iterable yielding text chunks, which suggests 
        # it might have been using a custom generate or I misread.
        # Let's check original code again: 
        # `response = self.model.generate(...)`
        # `for chunk in response: print(chunk) ...`
        # If `ipex_llm`'s `optimize_model` changes `generate` to return an iterator, then fine.
        # But standard transformers usage is via Streamer.
        # Let's stick to standard Streamer approach which is safer and likely supported.
        
        from transformers import TextIteratorStreamer
        from threading import Thread

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
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text
