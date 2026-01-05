import os
import sys

# ====================
# Logging Configuration
# ====================
LOGGING_ENABLED = int(os.getenv("LOGGING_ENABLED", "1"))
LOGGING_PREFIX = os.getenv("LOGGING_PREFIX", "")
LOGGING_CONFIG_PATH = os.getenv("LOGGING_CONFIG_PATH")
LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "INFO")
LOGGING_STREAM = os.getenv("LOGGING_STREAM", "ext://sys.stdout")
# Log retention and directory
LOGGING_RETENTION_DAYS = int(os.getenv("LOGGING_RETENTION_DAYS", "7"))
# Using abspath of this file (envs.py) -> (project root)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOGGING_DIR = os.getenv("LOGGING_DIR", os.path.join(PROJECT_ROOT, "logs"))

# ====================
# Storage / Vector DB Configuration
# ====================
# Base DB Path
DB_ROOT_PATH = os.getenv("DB_ROOT_PATH", os.path.join(PROJECT_ROOT, "db"))
DB_DATA_PATH = os.getenv("DB_DATA_PATH", os.path.join(DB_ROOT_PATH, "data"))

# SQLite Config
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", os.path.join(DB_DATA_PATH, "tasks.db"))

# Milvus Config
MILVUS_DATA_PATH = os.getenv("MILVUS_DATA_PATH", os.path.join(DB_DATA_PATH, "milvus", "milvus.db"))
MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_USER = os.getenv("MILVUS_USER", "")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "")
MILVUS_DB = os.getenv("MILVUS_DB", "default")
LLM_HISTORY_COLLECTION = os.getenv("LLM_HISTORY_COLLECTION", "llm_history")

# Embedding Config
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "/data/models/embeddings/seq/m3e-large")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
EMBEDDING_METRIC = os.getenv("EMBEDDING_METRIC", "IP")

# Qdrant Config
QDRANT_DATA_PATH = os.getenv("QDRANT_DATA_PATH", os.path.join(DB_ROOT_PATH, "qdrant", "data"))
QDRANT_HOST = os.getenv("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_URL = os.getenv("QDRANT_URL", "")

# Vector Store Type
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "milvus")

# ====================
# Gradio Configuration
# ====================
GRADIO_ANALYTICS_ENABLED = os.getenv("GRADIO_ANALYTICS_ENABLED", "False")
GRADIO_TEMP_DIR = os.getenv("GRADIO_TEMP_DIR", "storage")
GRADIO_ALLOWED_PATHS = os.getenv("GRADIO_ALLOWED_PATHS", "storage")
GRADIO_CACHE_EXAMPLES = os.getenv("GRADIO_CACHE_EXAMPLES", "False")
GRADIO_EXAMPLES_CACHE = os.getenv("GRADIO_EXAMPLES_CACHE", "pages/examples/")
GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
GRADIO_SHARE = os.getenv("GRADIO_SHARE", "False").lower() == "true"
GRADIO_DEBUG = os.getenv("GRADIO_DEBUG", "True").lower() == "true"

# ====================
# Other
# ====================
VLLM_HOST_IP = os.getenv("VLLM_HOST_IP", "")
VLLM_PORT = os.getenv("VLLM_PORT")
if VLLM_PORT:
    VLLM_PORT = int(VLLM_PORT)
else:
    VLLM_PORT = None
