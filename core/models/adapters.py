from __future__ import annotations

import importlib
from typing import Any, Iterable, List, Optional, Tuple, Dict

from .base import BaseModel
from .register import ModelFactory


def _camelize(*parts: str) -> str:
    return "".join(p.capitalize() for p in parts if p)


def resolve_model_config(domain: str, infer_arch: str, model_name: str) -> Tuple[Optional[str], str]:
    """
    Standardized mapping from configuration (arch, model_name) to (backend, impl_name).
    Replaces _map_backend_impl in servers.
    """
    backend = None
    impl = model_name.lower()

    # 1. Backend Mapping
    if infer_arch == "Pytorch":
        backend = "transformer"
    elif infer_arch == "OpenVino":
        backend = "openvino"
    elif infer_arch == "llama.cpp":
        backend = "llama_cpp"
    elif infer_arch == "ONNX":
        backend = "onnxruntime"

    # 2. Implementation Name Mapping (Normalization)
    if domain == "mt":
        name_map = {
            "Opus_mt_en_zh": "opus_mt",
            "Meta_NLLB": "meta_nllb",
            "Google_T5": "google_t5",
            "Ali_CSANMT": "ali_csanmt",
        }
        impl = name_map.get(model_name, impl)
    elif domain == "tts":
        name_map = {
            "ChatTTS": "chat_tts",
            "CosyVoice-300M": "cosy_voice",
            "Spark-TTS-0.5B": "spark_tts",
            "Fish-Speech-1.5": "fish_tts",
            "Kokoro-82M": "kokoro_tts",
            "F5-TTS": "f5_tts",
            "SpeechT5_TTS": "speecht5_tts",
            "Zonos-v0.1-transformer": "zonos_tts",
            "Dia-TTS": "dia_tts",
            "XTTS-v2": "x_tts",
            "ChatterBox": "chatterbox",
            "Index-TTS": "index_tts",
            "Oute-TTS": "oute_tts",
            "Vibe-Voice": "vibe_voice",
            "Voxcpm": "voxcpm",
        }
        impl = name_map.get(model_name, impl)
    elif domain == "asr":
        name_map = {
            "OpenAI_Whisper": "openai_whisper",
            "Ali_Paraformer": "ali_paraformer",
            "Meta_Wev2Vec_Conformer": "meta_wev2vec_conformer",
        }
        impl = name_map.get(model_name, impl)
    elif domain == "text2img":
        name_map = {
            "SD-1.5": "stable_diffusion",
            "SD-2": "stable_diffusion_2",
            "SD-XL": "stable_diffusion_xl",
            "SD-3": "stable_diffusion_3",
        }
        impl = name_map.get(model_name, impl)
    elif domain == "img2img":
        name_map = {
            "SD-1.5": "stable_diffusion",
            "SD-2": "stable_diffusion_2",
            "SD-XL": "stable_diffusion_xl",
            "SD-3": "stable_diffusion_3",
        }
        impl = name_map.get(model_name, impl)
    elif domain == "inpainting":
        name_map = {
            "SD-1.5": "stable_diffusion",
            "SD-2": "stable_diffusion_2",
            "SD-XL": "stable_diffusion_xl",
            "SD-3": "stable_diffusion_3",
        }
        impl = name_map.get(model_name, impl)
    elif domain == "llm":
        if impl == "llama3":
            impl = "llama"
        # DeepSeek, Qwen etc usually match lowercase or are handled by general rule

    return backend, impl


def _resolve_impl_path(domain: str, backend: Optional[str], impl: Optional[str]) -> Tuple[str, str]:
    backend = (backend or "").lower() or None
    impl = (impl or "").lower() or None

    # Special Class Name Mappings (for Acronyms or irregular casing)
    # Default _camelize handles snake_case -> CamelCase (e.g. cosy_voice -> CosyVoice)
    # This map handles exceptions (e.g. chat_tts -> ChatTTS, deepseek -> DeepSeek)
    special_class_names = {
        # LLM
        "deepseek": "DeepSeekModel",
        
        # ASR
        "openai_whisper": "OpenAIWhisperModel",
        
        # TTS
        "chat_tts": "ChatTTSModel",
        "spark_tts": "SparkTTSModel",
        "fish_tts": "FishTTSModel",
        "x_tts": "XTTSModel",
        "f5_tts": "F5TTSModel",
        "speecht5_tts": "SpeechT5TTSModel",
        "dia_tts": "DiaTTSModel",
        "index_tts": "IndexTTSModel",
        "oute_tts": "OuteTTSModel",
        "kokoro_tts": "KokoroTTSModel",

        # MT
        "opus_mt": "OpusMTModel",
        "meta_nllb": "MetaNLLBModel",
        "google_t5": "GoogleT5Model",
        "ali_csanmt": "AliCSANMTModel",

        # Text2Img
        "stable_diffusion": "ModelStableDiffusion",
        "stable_diffusion_2": "ModelStableDiffusion2",
        "stable_diffusion_xl": "ModelStableDiffusionXL",
        "stable_diffusion_3": "ModelStableDiffusion3",
    }

    # 已知映射（可持续扩充）
    known: dict[Tuple[str, Optional[str], Optional[str]], Tuple[str, str]] = {
        ("llm", "openvino", "llama"): (
            "core.models.sequences.llm.model_llama",
            "LlamaModel",
        ),
        ("llm", "openvino", "deepseek"): (
            "core.models.sequences.llm.model_deepseek",
            "DeepSeekModel",
        ),
        ("asr", None, "whisper"): (
            "core.models.audios.asr.model_openai_whisper",
            "OpenAIWhisperModel",
        ),
        ("tts", None, "spark_tts"): (
            "core.models.audios.tts.model_spark_tts",
            "SparkTTSModel",
        ),
    }
    key = (domain, backend, impl)
    if key in known:
        return known[key]

    # 约定式推断
    if domain == "llm":
        # Simplified mapping: map impl directly to model file, handle backend inside model
        # Old: core.models.sequences.llm.model_{impl}_{backend}
        # New: core.models.sequences.llm.model_{impl}
        
        # Special handling for legacy compatibility if files exist
        # But we prefer the unified model now
        
        # Check if it's a known unified model
        unified_models = ["deepseek", "llama", "qwen"]
        if impl in unified_models:
             mod_name = f"core.models.sequences.llm.model_{impl}"
             cls_name = special_class_names.get(impl, f"{_camelize(impl)}Model")
             return mod_name, cls_name
             
        # Fallback for others (like glm4v, minicpm which might not be unified yet)
        mod_name = f"core.models.sequences.llm.model_{impl}_{backend}"
        cls_name = f"{_camelize(impl)}{_camelize(backend)}Model"
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
        # Simplified mapping
        if impl == "transformer_sentence":
            return ("core.models.sequences.embedding.model_transformer_sentence", "TransformerSentenceModel")
        elif impl == "sentence_transformer":
            return ("core.models.sequences.embedding.model_sentence_transformer", "SentenceTransformerModel")
        
        mod_name = f"core.models.sequences.embedding.model_{impl}"
        cls_name = f"{_camelize(impl)}Model"
        return mod_name, cls_name
    elif domain == "audio_embedding":
        known_audio: dict[str, Tuple[str, str]] = {
            "clap": ("core.models.audios.retrieval.model_laion_clap", "LaionClapModel"),
            "msclap": ("core.models.audios.retrieval.model_microsoft_clap", "MicrosoftClapModel"),
            "clap_ipa": ("core.models.audios.retrieval.model_anyspeech_clap_ipa", "AnySpeechClapIpaModel"),
        }
        if impl in known_audio:
            return known_audio[impl]  # type: ignore[return-value]
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

    # 其他域可按需扩展 images/videos 等
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
            # Inject backend if accepted by load_model
            # Inspect signature to be safe, or just pass as kwargs if it accepts **kwargs
            # But BaseModel.load_model usually defined as (model_path, device) in abstract
            # However, python allows calling with more args if implementation supports it.
            # Our new models support backend arg.
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
