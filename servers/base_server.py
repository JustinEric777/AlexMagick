import abc
import time
import logging
from typing import Dict, Any, Optional, List, Union

from core.models.engine import ModelEngine
from core.models.adapters import resolve_model_config
from vectorstores.embedding import Embeddings

# Setup Logger
logger = logging.getLogger("BaseServer")


class BaseServer(abc.ABC):
    """
    Abstract Base Server Class.
    Implements:
    - Generic Model Management (init, reload, defaults)
    - Performance Metrics Helper
    """

    # Class attributes to be overridden by subclasses
    TASK_TYPE: str = ""
    MODEL_LIST: Dict[str, Any] = {}

    def __init__(self):
        # Model State
        self.task_type = self.TASK_TYPE
        self.model_list = self.MODEL_LIST
        self.engine: Optional[ModelEngine] = None
        self.pipeline: Any = None
        self.infer_arch: str = ""
        self.device: str = ""
        self.model_name: str = ""
        self.model_version_name: str = ""

    # --- Model Management (Refactored & Unified) ---

    def _resolve_defaults(self, infer_arch="", device="", model_name="", model_version=""):
        """Resolve missing parameters to defaults based on MODEL_LIST"""
        if not infer_arch or infer_arch not in self.MODEL_LIST:
            arch_list = list(self.MODEL_LIST.keys())
            infer_arch = arch_list[0] if arch_list else ""

        if not device:
            # Default to CPU for all models as per requirement
            device = "cpu"

        if not model_name or (infer_arch in self.MODEL_LIST and model_name not in self.MODEL_LIST[infer_arch]):
            model_list = list(self.MODEL_LIST.get(infer_arch, {}).keys())
            model_name = model_list[0] if model_list else ""

        if not model_version:
            versions = self.MODEL_LIST.get(infer_arch, {}).get(model_name, {}).get("model_list", [])
            model_version = versions[0] if versions else ""

        return infer_arch, device, model_name, model_version

    def _load_engine(self, infer_arch, device, model_name, model_version):
        """Core logic to load the model engine"""
        if not infer_arch or not model_name or infer_arch not in self.MODEL_LIST:
            logger.warning(f"Invalid model config: {infer_arch}/{model_name}")
            return

        model_info = self.MODEL_LIST[infer_arch].get(model_name)
        if not model_info:
            logger.warning(f"Model {model_name} not found in {infer_arch}")
            return

        # Ensure device is lowercase for PyTorch compatibility
        device = device.lower() if device else "cpu"

        model_path = model_info["model_path"]
        backend, impl = resolve_model_config(self.TASK_TYPE, infer_arch, model_name)
        logger.info(f"Loading model: {model_name} ({model_version}) on {device}...")

        self.engine = ModelEngine(
            model_type=self.TASK_TYPE,
            model_name_or_path=f"{model_path}/{model_version}",
            device=device,
            backend=backend,
            impl=impl,
        )
        self.pipeline = self.engine.model

        # Update state
        self.infer_arch = infer_arch
        self.device = device.lower()
        self.model_name = model_name
        self.model_version_name = model_version
        logger.info("Model loaded successfully.")

    def load_model(self, infer_arch: str = "", device: str = "", model_name: str = "", model_version: str = "", default: bool = False):
        """Reload model with new parameters"""
        if default:
            infer_arch, device, model_name, model_version = self._resolve_defaults()
        else:
             if not infer_arch or infer_arch not in self.MODEL_LIST:
                 logger.warning("No valid model configuration found. Skipping reload.")
                 return infer_arch, device, model_name, model_version
             
             # Resolve defaults for missing parts
             _, d, n, v = self._resolve_defaults(infer_arch, device, model_name, model_version)
             device = device or d
             model_name = model_name or n
             model_version = model_version or v

        self._load_engine(infer_arch, device, model_name, model_version)
        return self.infer_arch, self.device, self.model_name, self.model_version_name

    # --- Helper Methods ---

    def get_infer_arch_list(self):
        return list(self.MODEL_LIST.keys())

    def get_arch_model_list(self, infer_arch: Optional[str] = None):
        if not infer_arch or infer_arch not in self.MODEL_LIST:
            return []
        return list(self.MODEL_LIST[infer_arch].keys())

    def get_arch_device_list(self, infer_arch: Optional[str] = None):
        if not infer_arch or infer_arch not in self.MODEL_LIST:
            return []
        if infer_arch == "OpenVino":
            return ["cpu", "gpu", "npu", "auto"]
        return ["cpu", "cuda:0", "auto"]

    def get_model_list(self, infer_arch: Optional[str] = None, model_name: Optional[str] = None):
        if not infer_arch or not model_name:
            return []
        if infer_arch not in self.MODEL_LIST or model_name not in self.MODEL_LIST[infer_arch]:
            return []
        return self.MODEL_LIST[infer_arch][model_name].get("model_list", [])

    # --- Performance & Metric Helpers ---

    def pre_generate(self):
        """
        Check pipeline and start timer.
        Returns: start_time
        Raises: Exception if model not loaded
        """
        if not self.pipeline:
            raise RuntimeError(f"{self.TASK_TYPE} Model not loaded.")
        return time.time()

    def format_metric(self, metrics: Dict[str, Any]) -> str:
        """
        Format metrics dictionary into HTML string.
        """
        metric_lines = []
        for k, v in metrics.items():
            metric_lines.append(f"{k}：{v}")

        content = "<br>".join(metric_lines)
        return f"""<span style="color: red; display:block; float:right; margin-right: 10px; font-size: 14px;">
                {content}
                </span>"""

    def post_generate(self, start_time: float, extra_metrics: Dict[str, Any] = None) -> str:
        """
        Calculate cost_time and format metrics.
        Args:
            start_time: timestamp from pre_generate
            extra_metrics: specific metrics from the task (e.g. word count)
        Returns:
            HTML string of metrics
        """
        cost_time = round(time.time() - start_time, 3)

        metrics = {
            "model_name": self.model_name,
            "model_version": self.model_version_name,
            "cost_time": f"{cost_time}s",
            "device": self.device
        }
        if extra_metrics:
            metrics.update(extra_metrics)

        return self.format_metric(metrics)
    
    def safe_run(self, func, default=None, *args, **kwargs):
        """Safe execution wrapper with exception handling"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"{self.TASK_TYPE} Error: {e}", exc_info=True)
            return default

    @abc.abstractmethod
    def generate(self, **kwargs):
        """Core business logic, to be implemented by subclasses"""
        pass


class BaseImageGenServer(BaseServer):
    """
    Intermediate Base class for Image Generation Servers.
    (Text2Img, Img2Img, Inpainting)
    """
    
    def run_image_generation(self, *args, **kwargs):
        """
        Standard image generation flow:
        pre_generate -> pipeline.generate -> post_generate -> return output, metric
        """
        try:
            start_time = self.pre_generate()
            output, model_metrics = self.pipeline.generate(*args, **kwargs)
            metric = self.post_generate(start_time, model_metrics)
            return output, metric
        except Exception as e:
            logger.error(f"{self.TASK_TYPE} Generation Error: {e}", exc_info=True)
            return None, ""


class BaseEmbeddingServer(BaseServer, Embeddings):
    """
    Intermediate Base class for Embedding Servers (Text, Video).
    AudioEmbeddingServer has a different signature, so it might not inherit this directly
    or will override methods.
    """
    
    def generate(self, texts: str, search_text: str):
        """Common similarity calculation logic for demo"""
        try:
            start_time = self.pre_generate()
            assert len(texts) > 0 or len(search_text) > 0, "texts or search_text is empty"

            sentences = texts.split("\n")
            # Assuming pipeline has .encode() method which is standard for these
            texts_embeddings = self.pipeline.encode(sentences)
            search_embedding = self.pipeline.encode(search_text)

            scores = search_embedding @ texts_embeddings.T
            output = "\n".join(str(i) for i in scores[0])
            
            metric = self.post_generate(start_time, {"sentences_count": len(sentences)})
            return output, metric
        except Exception as e:
            logger.error(f"{self.TASK_TYPE} Generation Error: {e}", exc_info=True)
            return "", ""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.pipeline: return []
        return self.pipeline.encode(texts, self.model_name)

    def embed_query(self, text: str) -> List[float]:
        if not self.pipeline: return []
        return self.pipeline.encode(text, self.model_name)[0]
