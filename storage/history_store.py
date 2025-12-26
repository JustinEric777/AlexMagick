import json
import time
import uuid
from typing import Any, Dict, List

from vectorstores.factory import VectorStoreFactory
from vectorstores.embedding import SentenceTransformerEmbeddings


class HistoryStore:
    def __init__(self):
        self._emb = SentenceTransformerEmbeddings()
        self._store = VectorStoreFactory.create(embedding=self._emb)

    def save_llm_record(
        self,
        infer_arch: str,
        device: str,
        model_name: str,
        model_version: str,
        messages: List[Dict[str, Any]],
        output_text: str,
        metrics: Dict[str, Any],
    ) -> str:
        meta = {
            "infer_arch": infer_arch,
            "device": device,
            "model_name": model_name,
            "model_version": model_version,
            "messages": messages,
            "metrics": metrics,
        }
        rid = str(uuid.uuid4())
        self._store.add_texts([output_text], metadatas=[meta], ids=[rid])
        return rid

    def list_recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not getattr(self._store, "_connected", False) or self._store.collection is None:
            return []
        res = self._store.collection.query(expr="", output_fields=["id", "timestamp", "content", "metadata"], limit=limit)
        items = []
        for r in res:
            md = r.get("metadata")
            try:
                parsed = json.loads(md) if isinstance(md, str) and md.startswith("{") else md
            except Exception:
                parsed = {"raw": md}
            items.append(
                {
                    "id": r.get("id"),
                    "timestamp": r.get("timestamp"),
                    "content": r.get("content"),
                    "metadata": parsed,
                }
            )
        items.sort(key=lambda x: x["timestamp"], reverse=True)
        return items

