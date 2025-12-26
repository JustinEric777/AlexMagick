from __future__ import annotations

from typing import Any, Iterator
from threading import Thread

from transformers import AutoTokenizer, TextIteratorStreamer
from optimum.intel.openvino import OVModelForCausalLM, OVConfig

from .base import BackendRunner


class OpenVINOBackend(BackendRunner):
    def load_text_generation(self, model_name_or_path: str, **kwargs: Any) -> None:
        ov_config = kwargs.pop("ov_config", None) or OVConfig(device=self.device, dtype=kwargs.pop("dtype", "int8"))
        
        # Remove backend-specific kwargs that shouldn't go to from_pretrained if they exist
        # But here we assume kwargs are for from_pretrained
        
        self.model = OVModelForCausalLM.from_pretrained(
            model_name_or_path,
            use_cache=True,
            export=False,
            ov_config=ov_config,
            device=self.device,
            **kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)

    def generate_text(self, input_ids, **gen_kwargs: Any):
        return self.model.generate(input_ids=input_ids, **gen_kwargs)

    def stream_chat(self, messages: list[dict], **kwargs) -> Iterator[str]:
        # Similar logic to TorchBackend as OpenVINO model is HF compatible
        if not hasattr(self, "tokenizer") or not hasattr(self, "model"):
            raise RuntimeError("Model or Tokenizer not initialized.")
            
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Reset streamer for new generation
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        gen_kwargs = {
            "input_ids": input_ids,
            "streamer": self.streamer,
            "max_new_tokens": kwargs.get("max_tokens", 512),
            "do_sample": True,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        
        for new_text in self.streamer:
            yield new_text

