from __future__ import annotations

import time
import uuid
import os
from typing import Any, Iterable, List, Optional, Tuple, Dict

try:
    from pymilvus import MilvusClient, DataType
except ImportError:
    MilvusClient = None
    DataType = None

from vectorstores.base import VectorStore
from vectorstores.docstore.document import Document
from vectorstores.embedding import Embeddings
from config.storage_config import (
    MILVUS_DATA_PATH,
    LLM_HISTORY_COLLECTION,
    EMBEDDING_DIM,
    EMBEDDING_METRIC,
)


class Milvus(VectorStore):
    def __init__(self, collection_name: str = LLM_HISTORY_COLLECTION, embedding: Optional[Embeddings] = None):
        self.collection_name = collection_name
        self._embedding = embedding
        
        if MilvusClient is None:
             print("Warning: pymilvus not installed. History storage will be disabled.")
             self.client = None
             self._connected = False
             return

        try:
            # Create data directory if not exists (MilvusClient usually handles file creation but directory must exist)
            os.makedirs(os.path.dirname(MILVUS_DATA_PATH), exist_ok=True)
            
            self.client = MilvusClient(uri=MILVUS_DATA_PATH)
            self._ensure_collection(collection_name)
            self._connected = True
        except Exception as e:
            print(f"Warning: Failed to connect to Milvus Lite: {e}. History storage will be disabled.")
            self.client = None
            self._connected = False

    def _ensure_collection(self, name: str) -> None:
        if self.client.has_collection(name):
            return

        # Define schema explicitly to match previous structure
        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
            description="LLM history records"
        )
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field(field_name="timestamp", datatype=DataType.INT64)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=16384)
        schema.add_field(field_name="metadata", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="FLAT", # Milvus Lite local mode supports FLAT
            metric_type=EMBEDDING_METRIC,
            params={}
        )

        self.client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=index_params
        )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        if not self._connected:
            return []
        
        texts = list(texts)
        metadatas = metadatas or [{} for _ in texts]
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        ts = int(time.time() * 1000)
        
        if self._embedding is None:
            raise ValueError("Embeddings is required for Milvus.add_texts")
        
        vectors = self._embedding.embed_documents(texts)
        
        data = []
        for i, text in enumerate(texts):
            data.append({
                "id": ids[i],
                "timestamp": ts,
                "content": text,
                "metadata": str(metadatas[i]),
                "vector": vectors[i]
            })

        self.client.insert(collection_name=self.collection_name, data=data)
        return ids

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        if not self._connected:
            return False
        if ids is None or len(ids) == 0:
            return False
        
        # MilvusClient.delete takes filter string or list of ids if primary key is simple
        # For simple PK delete, we can pass ids list to 'filter' using 'in' operator
        # or use pks parameter if supported. 
        # delete(collection_name, ids=[...]) is supported in recent versions.
        # Let's use filter for safety or ids if available.
        # client.delete(collection_name, ids=ids) is standard.
        
        try:
            self.client.delete(collection_name=self.collection_name, ids=ids)
            return True
        except Exception:
            # Fallback to expression
            expr = f'id in {ids}'.replace("'", '"') # Milvus expressions use double quotes for strings usually? No, SQL style.
            # Actually standard python list str representation `['a', 'b']` works for `id in [...]` in Milvus.
            self.client.delete(collection_name=self.collection_name, filter=f"id in {ids}")
            return True

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embedding

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        if not self._connected:
            return []
        if self._embedding is None:
            raise ValueError("Embeddings is required for Milvus.similarity_search")
        
        qv = self._embedding.embed_query(query)
        
        res = self.client.search(
            collection_name=self.collection_name,
            data=[qv],
            anns_field="vector",
            search_params={"metric_type": EMBEDDING_METRIC, "params": {"ef": 128}},
            limit=k,
            output_fields=["id", "content", "metadata", "timestamp"]
        )
        
        docs: List[Document] = []
        # res is list of list of hits
        for hits in res:
            for hit in hits:
                # hit is a dict-like object in MilvusClient usually?
                # In MilvusClient, search returns list of list of dicts.
                entity = hit["entity"]
                md_raw = entity.get("metadata")
                docs.append(
                    Document(
                        page_content=entity.get("content"),
                        metadata={
                            "id": entity.get("id"),
                            "timestamp": entity.get("timestamp"),
                            "meta": md_raw,
                            "distance": float(hit["distance"]),
                        },
                    )
                )
        return docs

    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs: Any) -> List[Tuple[Document, float]]:
        docs = self.similarity_search(query, k=k, **kwargs)
        result: List[Tuple[Document, float]] = []
        for d in docs:
            result.append((d, d.metadata.get("distance", 0.0)))
        return result

    def _select_relevance_score_fn(self):
        return self._max_inner_product_relevance_score_fn

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = LLM_HISTORY_COLLECTION,
        **kwargs: Any,
    ) -> "Milvus":
        store = cls(collection_name=collection_name, embedding=embedding)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store
