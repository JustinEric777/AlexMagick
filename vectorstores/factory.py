from typing import Optional, Type, Dict
import logging

from vectorstores.base import VectorStore
from vectorstores.embedding import Embeddings
from vectorstores.milvus import Milvus
from vectorstores.qdrant import Qdrant
from vectorstores.faiss import FAISS
from config.storage_config import VECTOR_STORE_TYPE

logger = logging.getLogger(__name__)

class VectorStoreFactory:
    _registry: Dict[str, Type[VectorStore]] = {
        "milvus": Milvus,
        "qdrant": Qdrant,
        "faiss": FAISS
    }

    @classmethod
    def register(cls, name: str, store_cls: Type[VectorStore]):
        cls._registry[name] = store_cls

    @classmethod
    def create(cls, store_type: str = VECTOR_STORE_TYPE, embedding: Optional[Embeddings] = None, **kwargs) -> VectorStore:
        if store_type not in cls._registry:
            logger.warning(f"Unknown vector store type: {store_type}. Available: {list(cls._registry.keys())}. Falling back to Milvus.")
            store_type = "milvus"
        
        store_cls = cls._registry[store_type]
        return store_cls(embedding=embedding, **kwargs)
