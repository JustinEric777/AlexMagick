from typing import List
from config.text_embedding_config import MODEL_LIST, TASK_TYPE
from modules.servers.base_server import BaseServer, Metric
from modules.vectorstores.embedding import Embeddings


class VideoEmbeddingServer(BaseServer, Embeddings):
    def __init__(self):
        super().__init__()
        self.task_type = TASK_TYPE
        self.model_list = MODEL_LIST

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




