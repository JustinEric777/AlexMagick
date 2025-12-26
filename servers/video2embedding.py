from typing import List
from config.text_embedding_config import MODEL_LIST, TASK_TYPE
from servers.base_server import BaseServer, Metric
from vectorstores.embedding import Embeddings
from core.models.engine import ModelEngine
from core.models.adapters import resolve_model_config


class Video2EmbeddingServer(BaseServer, Embeddings):
    def __init__(self):
        super().__init__()
        self.task_type = TASK_TYPE
        self.model_list = MODEL_LIST
        self.engine = None

    def init_model(self, params: dict):
        arch = params.get("infer_arch", "")
        device = params.get("device", "")
        model_name = params.get("model_name", "")
        version = params.get("model_version", "")

        if params.get("task_type") != self.task_type or arch not in self.model_list or model_name not in self.model_list[arch]:
            arch_list = self.get_infer_arch_list()
            arch = arch_list[0] if len(arch_list) > 0 else ""
            
            device_list = self.get_arch_device_list(infer_arch=arch)
            device = device_list[0] if len(device_list) > 0 else ""
            
            model_name_list = self.get_arch_model_list(infer_arch=arch)
            model_name = model_name_list[0] if len(model_name_list) > 0 else ""
            
            version_list = self.get_model_list(infer_arch=arch, model_name=model_name)
            version = version_list[0] if len(version_list) > 0 else ""

        model_path = self.model_list[arch][model_name]["model_path"]
        backend, impl = resolve_model_config("video_embedding", arch, model_name)
        self.engine = ModelEngine(
            model_type="video_embedding",
            model_name_or_path=f"{model_path}/{version}",
            device=device,
            backend=backend,
            impl=impl,
        )
        self.pipeline = self.engine.model
        self.infer_arch = arch
        self.device = device
        self.model_name = model_name
        self.model_version_name = version

    def reload_model(self, infer_arch: str = "", device: str = "", model_name: str = "", model_version: str = "", default: bool = False):
        if default:
            arch_list = self.get_infer_arch_list()
            infer_arch = arch_list[0] if len(arch_list) > 0 else ""
            device_list = self.get_arch_device_list(infer_arch)
            device = device_list[0] if len(device_list) > 0 else ""
            model_name_list = self.get_arch_model_list(infer_arch)
            model_name = model_name_list[0] if len(model_name_list) > 0 else ""
            model_version_list = self.get_model_list(infer_arch, model_name)
            model_version = model_version_list[0] if len(model_version_list) > 0 else ""
        model_path = self.model_list[infer_arch][model_name]["model_path"]
        backend, impl = resolve_model_config("video_embedding", infer_arch, model_name)
        self.engine = ModelEngine(
            model_type="video_embedding",
            model_name_or_path=f"{model_path}/{model_version}",
            device=device,
            backend=backend,
            impl=impl,
        )
        self.pipeline = self.engine.model
        self.infer_arch = infer_arch
        self.device = device
        self.model_name = model_name
        self.model_version_name = model_version
        return infer_arch, device, model_name, model_version

    @Metric()
    def generate(self, texts: str, search_text: str, model_name: str):
        assert len(texts) > 0 or len(search_text) > 0, "texts or search_text is empty"

        sentences = texts.split("\n")
        texts_embeddings = self.pipeline.encode(sentences, model_name)
        search_embedding = self.pipeline.encode(search_text, model_name)

        scores = search_embedding @ texts_embeddings.T
        return "\n".join(str(i) for i in scores[0])

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.pipeline.encode(texts, self.model_name)

    def embed_query(self, text: str) -> List[float]:
        return self.pipeline.encode(text, self.model_name)[0]


