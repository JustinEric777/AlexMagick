from config.text_embedding_config import MODEL_LIST, TASK_TYPE
from modules.servers.base_server import BaseServer, Metric


class TextEmbeddingServer(BaseServer):
    def __init__(self):
        super().__init__()
        self.task_type = TASK_TYPE
        self.model_list = MODEL_LIST

    @Metric()
    def generate(self, texts: str, search_text: str, model_name: str):
        assert len(texts) > 0 or len(search_text) > 0, "texts or search_text is empty"

        sentences = texts.split("\n")
        texts_embeddings = self.pipeline_object.encode(sentences, model_name)
        search_embedding = self.pipeline_object.encode(search_text, model_name)

        scores = search_embedding @ texts_embeddings.T
        return "\n".join(str(i) for i in scores[0])



