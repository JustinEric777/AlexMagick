from __future__ import annotations

from typing import Any, Iterator
from threading import Thread

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TextIteratorStreamer

from .base import BackendRunner


class TorchBackend(BackendRunner):
    def load_text_generation(self, model_name_or_path: str, **kwargs: Any) -> None:
        # Default dtype logic if not provided in kwargs
        if "torch_dtype" not in kwargs:
            kwargs["torch_dtype"] = torch.float16 if (kwargs.pop("fp16", True) and torch.cuda.is_available()) else torch.float32
        
        # Handling specific kwargs that might conflict or need separate handling
        trust_remote_code = kwargs.get("trust_remote_code", False)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **kwargs
        )
        
        # Try to load generation config
        try:
            self.model.generation_config = GenerationConfig.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code
            )
        except Exception:
            pass

        # If device_map is not auto/cpu, we might need manual move (though from_pretrained usually handles it)
        if "device_map" not in kwargs and self.device != "cpu":
             self.model.to(self.device)

    def generate_text(self, input_ids, **gen_kwargs: Any):
        return self.model.generate(input_ids=input_ids, **gen_kwargs)

    def stream_chat(self, messages: list[dict], **kwargs) -> Iterator[str]:
        if not hasattr(self, "tokenizer") or not hasattr(self, "model"):
            raise RuntimeError("Model or Tokenizer not initialized.")
        
        # Prepare inputs
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Setup streamer
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Prepare generation kwargs
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
        
        # Run generation in a separate thread
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        
        for new_text in streamer:
            yield new_text

