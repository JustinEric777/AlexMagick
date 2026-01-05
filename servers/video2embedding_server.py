from config.video_embedding_config import MODEL_LIST, TASK_TYPE
from servers.base_server import BaseEmbeddingServer


class Video2EmbeddingServer(BaseEmbeddingServer):
    TASK_TYPE = TASK_TYPE
    MODEL_LIST = MODEL_LIST

    def __init__(self):
        super().__init__()
