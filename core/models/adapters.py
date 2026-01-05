from __future__ import annotations

import importlib
from typing import Any, Iterable, List, Optional, Tuple, Dict

from core.models.base import BaseModel
from core.models.register import ModelFactory


def _camelize(*parts: str) -> str:
    return "".join(p.capitalize() for p in parts if p)


def resolve_model_config(domain: str, infer_arch: str, model_name: str) -> Tuple[Optional[str], str]:
    """
    Standardized mapping from configuration (arch, model_name) to (backend, impl_name).
    Replaces _map_backend_impl in servers.
    """
    backend = None
    
    # 1. Backend Mapping
    if infer_arch == "Pytorch":
        backend = "transformer"
    elif infer_arch == "OpenVino":
        backend = "openvino"
    elif infer_arch == "llama.cpp":
        backend = "llama_cpp"
    elif infer_arch == "ONNX":
        backend = "onnxruntime"

    # 2. Dynamic Config Lookup
    try:
        # Import config module dynamically
        config_module = importlib.import_module(f"config.{domain}_config")
        model_list = getattr(config_module, "MODEL_LIST", {})
        
        if infer_arch in model_list and model_name in model_list[infer_arch]:
            conf = model_list[infer_arch][model_name]
            path = conf.get("model_provider_path")
            name = conf.get("model_provider_name")
            
            if path and name:
                # Encode path and name into impl string "path:name"
                return backend, f"{path}:{name}"
    except (ImportError, AttributeError):
        pass

    # Fallback to default naming convention if not found in config
    return backend, model_name.lower()


def _resolve_impl_path(domain: str, backend: Optional[str], impl: Optional[str]) -> Tuple[str, str]:
    # 1. Check for direct path:class format (from resolve_model_config) - Case sensitive!
    if impl and ":" in impl:
        return impl.split(":", 1)

    backend = (backend or "").lower() or None
    impl = (impl or "").lower() or None

    # 2. Fallback Logic (Legacy Support)
    # Special Class Name Mappings
    special_class_names = {
        # LLM
        "deepseek": "DeepSeekModel",
        "chat_tts": "ChatTTSModel",
        # Add others if needed for fallback, but ideally everything is in config now
    }
    
    # 约定式推断
    if domain == "llm":
        mod_name = f"core.models.sequences.llm.model_{impl}"
        cls_name = special_class_names.get(impl, f"{_camelize(impl)}Model")
        return mod_name, cls_name
    elif domain == "asr":
        mod_name = f"core.models.audios.asr.model_{impl}"
        cls_name = special_class_names.get(impl, f"{_camelize(impl)}Model")
        return mod_name, cls_name
    elif domain == "tts":
        mod_name = f"core.models.audios.tts.model_{impl}"
        cls_name = special_class_names.get(impl, f"{_camelize(impl)}Model")
        return mod_name, cls_name
    elif domain == "mt":
        mod_name = f"core.models.sequences.mt.model_{impl}"
        cls_name = special_class_names.get(impl, f"{_camelize(impl)}Model")
        return mod_name, cls_name
    elif domain == "text_embedding":
        mod_name = f"core.models.sequences.embedding.model_{impl}"
        cls_name = f"{_camelize(impl)}Model"
        return mod_name, cls_name
    elif domain == "audio_embedding":
        mod_name = f"core.models.audios.retrieval.model_{impl}"
        cls_name = f"{_camelize(impl)}Model"
        return mod_name, cls_name
    elif domain == "text2img":
        mod_name = f"core.models.images.text2img.model_{impl}"
        cls_name = special_class_names.get(impl, f"{_camelize(impl)}Model")
        return mod_name, cls_name
    elif domain == "img2img":
        mod_name = f"core.models.images.img2img.model_{impl}"
        cls_name = special_class_names.get(impl, f"{_camelize(impl)}Model")
        return mod_name, cls_name
    elif domain == "inpainting":
        mod_name = f"core.models.images.inpainting.model_{impl}"
        cls_name = special_class_names.get(impl, f"{_camelize(impl)}Model")
        return mod_name, cls_name
    elif domain == "video_embedding":
        mod_name = f"core.models.videos.embedding.model_{impl}"
        cls_name = f"{_camelize(impl)}Model"
        return mod_name, cls_name
    elif domain == "multimodal_llm":
        mod_name = f"core.models.multimodals.llm.model_{impl}_{backend}"
        cls_name = f"{_camelize(impl)}{_camelize(backend)}Model"
        return mod_name, cls_name

    raise ValueError(f"Unsupported domain for resolution: {domain}")


class _ExternalWrapper:
    """对外部实现实例做一个统一的方法名适配。"""

    def __init__(self, impl_obj: Any):
        self._impl = impl_obj

    def call(self, candidates: List[str], *args, **kwargs):
        for name in candidates:
            if hasattr(self._impl, name):
                return getattr(self._impl, name)(*args, **kwargs)
        raise AttributeError(f"None of methods {candidates} found in {type(self._impl).__name__}")


class _UnifiedAdapter(BaseModel):
    """统一适配器：根据 domain/backends/impl 加载 modules 下的实现。"""

    DOMAIN: str = ""

    def __init__(self, *args, backend: Optional[str] = None, impl: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.backend = backend
        self.impl = impl
        self._wrapper: Optional[_ExternalWrapper] = None

    def load(self) -> None:
        mod_path, cls_name = _resolve_impl_path(self.DOMAIN, self.backend, self.impl)
        module = importlib.import_module(mod_path)
        impl_cls = getattr(module, cls_name)
        obj = impl_cls()
        # modules 下约定名通常为 load_model(model_path, device)
        if hasattr(obj, "load_model"):
            try:
                obj.load_model(self.model_name_or_path or "", self.device, backend=self.backend)
            except TypeError:
                # Fallback for old models not accepting backend arg
                obj.load_model(self.model_name_or_path or "", self.device)
        else:
            # 兜底：若存在 load()
            if hasattr(obj, "load"):
                obj.load()
        self._wrapper = _ExternalWrapper(obj)

    def close(self) -> None:
        if self._wrapper is not None and hasattr(self._wrapper._impl, "release"):
            self._wrapper._impl.release()
        self._wrapper = None


@ModelFactory.register("llm")
class UnifiedLLMAdapter(_UnifiedAdapter):
    DOMAIN = "llm"

    def generate(self, inputs: Iterable[dict] | List[dict] | str, **kwargs: Any):
        assert self._wrapper is not None
        # 优先走 chat 接口；否则尝试 generate/inference
        if isinstance(inputs, (list, tuple)):
            return self._wrapper.call(["chat", "generate", "inference"], inputs, **kwargs)
        return self._wrapper.call(["generate", "inference", "chat"], inputs, **kwargs)

    def chat(self, history, *args, **kwargs):
        assert self._wrapper is not None
        return self._wrapper.call(["chat"], history, *args, **kwargs)


@ModelFactory.register("asr")
class UnifiedASRAdapter(_UnifiedAdapter):
    DOMAIN = "asr"

    def generate(self, inputs: str, **kwargs: Any) -> Any:
        assert self._wrapper is not None
        # 常见为 generate(audio) 或 inference(audio)
        return self._wrapper.call(["generate", "inference"], inputs, **kwargs)


@ModelFactory.register("tts")
class UnifiedTTSAdapter(_UnifiedAdapter):
    DOMAIN = "tts"

    def generate(self, inputs: str, **kwargs: Any) -> Any:
        assert self._wrapper is not None
        # 常见为 inference(text, **kwargs)
        return self._wrapper.call(["inference", "generate"], inputs, **kwargs)


@ModelFactory.register("text_embedding")
class UnifiedTextEmbeddingAdapter(_UnifiedAdapter):
    DOMAIN = "text_embedding"

    def generate(self, inputs: Iterable[str] | str, **kwargs: Any) -> Any:
        assert self._wrapper is not None
        return self._wrapper.call(["encode"], inputs, **kwargs)

    def encode(self, inputs: Iterable[str] | str, *args, **kwargs):
        assert self._wrapper is not None
        return self._wrapper.call(["encode"], inputs, *args, **kwargs)


@ModelFactory.register("mt")
class UnifiedMTAdapter(_UnifiedAdapter):
    DOMAIN = "mt"

    def generate(self, inputs: str, **kwargs: Any) -> Any:
        assert self._wrapper is not None
        return self._wrapper.call(["translate", "generate", "inference"], inputs, **kwargs)


@ModelFactory.register("audio_embedding")
class UnifiedAudioEmbeddingAdapter(_UnifiedAdapter):
    DOMAIN = "audio_embedding"

    def generate(self, inputs: Any, **kwargs: Any) -> Any:
        assert self._wrapper is not None
        return self._wrapper.call(["calculate_sim"], **kwargs)

    def get_text_features(self, texts: Any, *args, **kwargs):
        assert self._wrapper is not None
        return self._wrapper.call(["get_text_features"], texts, *args, **kwargs)

    def get_audio_features(self, audios: Any, *args, **kwargs):
        assert self._wrapper is not None
        return self._wrapper.call(["get_audio_features"], audios, *args, **kwargs)

    def calculate_sim(self, *args, **kwargs):
        assert self._wrapper is not None
        return self._wrapper.call(["calculate_sim"], *args, **kwargs)


@ModelFactory.register("text2img")
class UnifiedText2ImgAdapter(_UnifiedAdapter):
    DOMAIN = "text2img"

    def generate(self, *args, **kwargs):
        assert self._wrapper is not None
        return self._wrapper.call(["generate"], *args, **kwargs)


@ModelFactory.register("img2img")
class UnifiedImg2ImgAdapter(_UnifiedAdapter):
    DOMAIN = "img2img"

    def generate(self, *args, **kwargs):
        assert self._wrapper is not None
        return self._wrapper.call(["generate"], *args, **kwargs)


@ModelFactory.register("inpainting")
class UnifiedInpaintingAdapter(_UnifiedAdapter):
    DOMAIN = "inpainting"

    def generate(self, *args, **kwargs):
        assert self._wrapper is not None
        return self._wrapper.call(["generate"], *args, **kwargs)


@ModelFactory.register("video_embedding")
class UnifiedVideoEmbeddingAdapter(_UnifiedAdapter):
    DOMAIN = "video_embedding"

    def generate(self, *args, **kwargs):
        assert self._wrapper is not None
        return self._wrapper.call(["encode"], *args, **kwargs)

    def encode(self, *args, **kwargs):
        assert self._wrapper is not None
        return self._wrapper.call(["encode"], *args, **kwargs)


@ModelFactory.register("multimodal_llm")
class UnifiedMultimodalLLMAdapter(_UnifiedAdapter):
    DOMAIN = "multimodal_llm"

    def generate(self, *args, **kwargs):
        assert self._wrapper is not None
        # 优先走 chat 接口
        return self._wrapper.call(["chat", "generate"], *args, **kwargs)

    def chat(self, *args, **kwargs):
        assert self._wrapper is not None
        return self._wrapper.call(["chat"], *args, **kwargs)
