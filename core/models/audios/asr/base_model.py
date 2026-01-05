import abc
from typing import Any, List, Union, Tuple, Dict
from core.models.loader import BaseModel as LoaderBaseModel
from core.monitor.system import SystemMonitor

class BaseModel(LoaderBaseModel):
    pipeline: Any = None
    dtype: Any = None

    def generate(self, audio: str) -> Tuple[str, Dict[str, Any]]:
        tracker = SystemMonitor.Tracker()
        with tracker:
            result = self._generate(audio)
        
        stats = tracker.get_stats()
        metrics = {
            "cpu_usage": stats["peak_cpu"],
            "memory_usage": stats["peak_mem"],
            "avg_cpu": stats["avg_cpu"],
            "avg_mem": stats["avg_mem"]
        }
        return result, metrics

    @abc.abstractmethod
    def _generate(self, audio: str) -> str:
        pass
