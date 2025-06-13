from typing import List, Union
from config.audio_embedding_config import MODEL_LIST, TASK_TYPE
from modules.servers.base_server import BaseServer, Metric
from modules.vectorstores.embedding import Embeddings


class AudioEmbeddingServer(BaseServer, Embeddings):
    def __init__(self):
        super().__init__()
        self.task_type = TASK_TYPE
        self.model_list = MODEL_LIST

    @Metric()
    def generate(self, texts: Union[str, List[str]], audios: Union[str, List[str]], model_name: str):
        assert len(texts) > 0 or len(audios) > 0, "texts or audios is empty"
        sentences = texts.split("\n")

        results = self.pipeline.calculate_sim(texts=sentences, audios=audios)

        return "\n".join(str(i) for i in results[0])

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.pipeline.get_audio_features(texts, self.model_name)

    def embed_query(self, text: str) -> List[float]:
        return self.pipeline.get_text_features(text, self.model_name)[0]




