from typing import List

from config.video_embedding_config import MODEL_LIST, TASK_TYPE
from servers.base_server import BaseServer
from vectorstores.embedding import Embeddings


class Video2EmbeddingServer(BaseServer, Embeddings):
    TASK_TYPE = TASK_TYPE
    MODEL_LIST = MODEL_LIST

    def __init__(self):
        super().__init__()

    def generate(self, texts: str, search_text: str, model_name: str):
        try:
            self.pre_generate()
            assert len(texts) > 0 or len(search_text) > 0, "texts or search_text is empty"

            sentences = texts.split("\n")
            texts_embeddings = self.pipeline.encode(sentences, model_name)
            search_embedding = self.pipeline.encode(search_text, model_name)

            scores = search_embedding @ texts_embeddings.T
            return "\n".join(str(i) for i in scores[0])
        except Exception as e:
            print(f"VideoEmbedding Error: {e}")
            return ""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.pipeline: return []
        return self.pipeline.encode(texts, self.model_name)

    def embed_query(self, text: str) -> List[float]:
        if not self.pipeline: return []
        return self.pipeline.encode(text, self.model_name)[0]
