from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ModelContext:
    """模型构建上下文，统一参数传递。"""
    model_type: str
    model_name_or_path: Optional[str] = None
    device: str = "cpu"
    batch_size: int = 1
    extra_options: Dict[str, Any] = field(default_factory=dict)
