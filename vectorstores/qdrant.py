from __future__ import annotations

import uuid
from typing import Any, Iterable, List, Optional

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest
except ImportError:
    QdrantClient = None
    rest = None

from vectorstores.base import VectorStore
from vectorstores.docstore.document import Document
from vectorstores.embedding import Embeddings
from config.storage_config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_API_KEY,
    QDRANT_URL,
    LLM_HISTORY_COLLECTION,
    EMBEDDING_DIM,
    EMBEDDING_METRIC,
)


class Qdrant(VectorStore):
    def __init__(
        self,
        collection_name: str = LLM_HISTORY_COLLECTION,
        embedding: Optional[Embeddings] = None,
        url: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        api_key: Optional[str] = None,
    ):
        if QdrantClient is None:
            raise ImportError("Could not import qdrant-client. Please install it with `pip install qdrant-client`.")

        self.collection_name = collection_name
        self._embedding = embedding

        # Priority: explicit arg > config > default
        self.url = url or QDRANT_URL
        self.host = host or QDRANT_HOST
        self.port = port or QDRANT_PORT
        self.api_key = api_key or QDRANT_API_KEY

        try:
            if self.url:
                self.client = QdrantClient(url=self.url, api_key=self.api_key)
            else:
                self.client = QdrantClient(host=self.host, port=self.port, api_key=self.api_key)

            self._ensure_collection()
            self._connected = True
        except Exception as e:
            print(f"Warning: Failed to connect to Qdrant: {e}. Storage will be disabled.")
            self.client = None
            self._connected = False

    def _ensure_collection(self):
        # Check if collection exists
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            # Map metric
            if EMBEDDING_METRIC.upper() == "IP":
                distance = rest.Distance.DOT
            elif EMBEDDING_METRIC.upper() == "L2":
                distance = rest.Distance.EUCLID
            else:
                distance = rest.Distance.COSINE

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=distance,
                ),
            )

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        if not self._connected:
            return []

        if self._embedding is None:
            raise ValueError("Embeddings is required for Qdrant.add_texts")

        texts = list(texts)
        metadatas = metadatas or [{} for _ in texts]
        ids = ids or [str(uuid.uuid4()) for _ in texts]

        embeddings = self._embedding.embed_documents(texts)

        points = []
        for i, text in enumerate(texts):
            payload = metadatas[i].copy()
            payload["page_content"] = text
            # Ensure metadata values are compatible (Qdrant handles primitives well)

            points.append(
                rest.PointStruct(
                    id=ids[i],
                    vector=embeddings[i],
                    payload=payload
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        return ids

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        if not self._connected or not ids:
            return False

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=rest.PointIdsList(points=ids)
        )
        return True

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        if not self._connected:
            return []

        if self._embedding is None:
            raise ValueError("Embeddings is required for Qdrant.similarity_search")

        query_vector = self._embedding.embed_query(query)

        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k,
            **kwargs
        )

        docs = []
        for hit in search_result:
            payload = hit.payload
            page_content = payload.pop("page_content", "")

            # Reconstruct metadata (everything else in payload)
            # Add implicit metadata
            metadata = payload
            metadata["id"] = hit.id
            metadata["score"] = hit.score

            docs.append(Document(page_content=page_content, metadata=metadata))

        return docs

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = LLM_HISTORY_COLLECTION,
        **kwargs: Any,
    ) -> "Qdrant":
        qdrant = cls(collection_name=collection_name, embedding=embedding, **kwargs)
        qdrant.add_texts(texts, metadatas=metadatas, ids=ids)
        return qdrant
