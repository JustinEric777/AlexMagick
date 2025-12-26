from abc import ABC, abstractmethod
from typing import List
import os
from core.models.engine import ModelEngine
from config.text_embedding_config import MODEL_LIST


class Embeddings(ABC):
    """Interface for embedding models."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""


class SentenceTransformerEmbeddings(Embeddings):
    """
    Wrapper around core.models.sequences.embedding.model_sentence_transformer
    to provide compatible interface for HistoryStore.
    """
    def __init__(self, model_name: str = "m3e-large", device: str = "cpu"):
        self.model_name = model_name
        self.engine = None
        
        try:
            if "Pytorch" not in MODEL_LIST or "sentence_transformer" not in MODEL_LIST["Pytorch"]:
                print(f"Info: History embedding model configuration not found. History storage will be disabled.")
                self.model = None
                return

            base_path = MODEL_LIST["Pytorch"]["sentence_transformer"]["model_path"]
            model_path = os.path.join(base_path, model_name)
            
            self.engine = ModelEngine(
                model_type="text_embedding",
                model_name_or_path=model_path,
                device=device,
                backend="transformer", # backend
                impl="sentence_transformer" # impl
            )
            self.model = self.engine.model
        except Exception as e:
            print(f"Warning: Failed to load History embedding model {model_name}: {e}")
            self.model = None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.model:
            return self.model.encode(texts, model_name=self.model_name, return_numpy=False).tolist()
        return [[0.0]*768 for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        if self.model:
            res = self.model.encode([text], model_name=self.model_name, return_numpy=False)
            return res[0].tolist()
        return [0.0]*768

