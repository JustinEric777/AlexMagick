from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # log
    LOGGING_ENABLED: int = 1
    LOGGING_PREFIX: str = ""
    LOGGING_CONFIG_PATH: str | None = None
    LOGGING_LEVEL: str = "INFO"
    LOGGING_STREAM: str = "ext://sys.stdout"

    # db
    VLLM_HOST_IP: str = ""
    VLLM_PORT: int | None = None
