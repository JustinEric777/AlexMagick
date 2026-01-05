import logging
from typing import List, Union

from config.audio_embedding_config import MODEL_LIST, TASK_TYPE
from servers.base_server import BaseServer
from vectorstores.embedding import Embeddings

logger = logging.getLogger(__name__)

class AudioEmbeddingServer(BaseServer, Embeddings):
    TASK_TYPE = TASK_TYPE
    MODEL_LIST = MODEL_LIST

    def __init__(self):
        super().__init__()

    def generate(self, texts: Union[str, List[str]], audios: Union[str, List[str]]):
        try:
            start_time = self.pre_generate()
            
            sentences = texts.split("\n") if isinstance(texts, str) else texts
            results = self.pipeline.calculate_sim(texts=sentences, audios=audios)
            
            # Simple result return, no complex metrics for this one in original code
            return "\n".join(str(i) for i in results[0])
        except Exception as e:
            logger.error(f"AudioEmbedding Error: {e}", exc_info=True)
            return ""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.pipeline: return []
        return self.pipeline.get_audio_features(texts, self.model_name)

    def embed_query(self, text: str) -> List[float]:
        if not self.pipeline: return []
        return self.pipeline.get_text_features(text, self.model_name)[0]
