from __future__ import annotations

import gc
import torch
from typing import Any, Iterable, Optional

from ..logger.logger import get_logger
from .context import ModelContext
from .register import ModelFactory


@get_logger("ModelEngine")
class ModelEngine:
    """统一的模型加载与推理引擎，接口贴近 vLLM 用户体验。

    用法：
        engine = ModelEngine(model_type="llm", model_name_or_path="Qwen/Qwen2")
        outputs = engine.generate(["你好", "介绍一下你自己"])
    """
    _current_engine: Optional['ModelEngine'] = None

    def __init__(
        self,
        model_type: str,
        model_name_or_path: Optional[str] = None,
        device: str = "cpu",
        batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        # Ensure only one model is loaded at a time
        if ModelEngine._current_engine is not None:
            self.logger.info("Releasing previous model to free memory...")
            ModelEngine._current_engine.close()
            ModelEngine._current_engine = None

        self.context = ModelContext(
            model_type=model_type,
            model_name_or_path=model_name_or_path,
            device=device,
            batch_size=batch_size,
            extra_options=kwargs,
        )
        self.model = ModelFactory.create(self.context)
        ModelEngine._current_engine = self
        
        self.logger.info(
            "model initialized: type=%s, name_or_path=%s, device=%s",
            model_type,
            model_name_or_path,
            device,
        )

    def generate(self, inputs: Iterable[str] | str, **kwargs: Any):
        return self.model.generate(inputs, **kwargs)

    def close(self) -> None:
        try:
            if hasattr(self, 'model') and self.model:
                self.model.close()
        except Exception as e:
            self.logger.error(f"Error closing model: {e}")
        finally:
            # Force cleanup
            if hasattr(self, 'model'):
                del self.model
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.ipc_collect()
            
            self.logger.info("model closed and memory released")
            
            if ModelEngine._current_engine == self:
                ModelEngine._current_engine = None


