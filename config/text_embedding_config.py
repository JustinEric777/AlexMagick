from typing import Dict, Any

TASK_TYPE = "sequence-text2embedding"

MODEL_LIST: Dict[str, Any] = {
    "Pytorch": {
        "transformer_sentence": {
            "model_provider_path": "modules.models.sequences.embedding.model_transformer_sentence",
            "model_provider_name": "TransformerSentenceModel",
            "model_path": "/data/models/embeddings/seq",
            "model_list": [
                "bge-base-zh-v1.5",
                "bce-embedding-base_v1",
                "360Zhinao-search",
                "gte-multilingual-base",
                "gte-base-zh",
                "gte-small-zh",
            ]
        },
        "sentence_transformer": {
            "model_provider_path": "modules.models.sequences.embedding.model_sentence_transformer",
            "model_provider_name": "SentenceTransformerModel",
            "model_path": "/data/models/embeddings/seq",
            "model_list": [
                "m3e-large",
                "jina-embeddings-v2-base-zh",
                "Yinka",
                "Dmeta-embedding-zh",
                "zpoint_large_embedding_zh"
                "Qwen3-Embedding-0.6B"
            ]
        },
    }
}
