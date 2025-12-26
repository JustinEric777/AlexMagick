import os

# Base DB Path
DB_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db")

# Milvus Config
MILVUS_DATA_PATH = os.path.join(DB_ROOT_PATH, "milvus", "data", "milvus.db")
# Kept for backward compatibility if needed, but primary use is now Lite
MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_USER = os.getenv("MILVUS_USER", "")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "")
MILVUS_DB = os.getenv("MILVUS_DB", "default")

LLM_HISTORY_COLLECTION = os.getenv("LLM_HISTORY_COLLECTION", "llm_history")

EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "/data/models/embeddings/seq/m3e-large")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
EMBEDDING_METRIC = os.getenv("EMBEDDING_METRIC", "IP")

# Qdrant Config
QDRANT_DATA_PATH = os.path.join(DB_ROOT_PATH, "qdrant", "data")
QDRANT_HOST = os.getenv("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_URL = os.getenv("QDRANT_URL", "")  # Optional: full URL

# Configurable vector store type: milvus, qdrant, faiss
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "milvus")
