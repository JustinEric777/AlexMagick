from __future__ import annotations

from typing import Any, List, Union, Iterator

from llama_cpp import Llama

from .base import BackendRunner


class LlamaCppBackend(BackendRunner):
    def load_text_generation(self, model_name_or_path: str, **kwargs: Any) -> None:
        self.model = Llama(model_path=model_name_or_path, **kwargs)
        self.tokenizer = None  # llama.cpp manages its own tokenization usually

    def generate_text(self, input_ids, **gen_kwargs: Any):
        # Fallback to simple generation if needed, but prefer create_chat_completion
        prompt = gen_kwargs.pop("prompt")
        out = self.model(prompt=prompt, **gen_kwargs)
        return out["choices"][0]["text"]
    
    def create_chat_completion(self, messages: List[dict], **kwargs) -> Union[dict, Iterator[dict]]:
        return self.model.create_chat_completion(messages=messages, **kwargs)
    
    def __call__(self, *args, **kwargs):
        # Allow direct call for raw generation
        return self.model(*args, **kwargs)

    def stream_chat(self, messages: list[dict], **kwargs) -> Iterator[str]:
        response = self.create_chat_completion(
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 512),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            stream=True
        )
        for chunk in response:
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                yield delta["content"]


