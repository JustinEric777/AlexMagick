import abc
from typing import Any
from core.models.loader import BaseModel as LoaderBaseModel


from core.monitor.system import SystemMonitor
from typing import Any, Tuple, Dict

class BaseModel(LoaderBaseModel):
    def generate(self, text: str) -> Tuple[str, Dict[str, Any]]:
        tracker = SystemMonitor.Tracker()
        with tracker:
            result = self.translate(text)
        
        stats = tracker.get_stats()
        metrics = {
            "cpu_usage": stats["peak_cpu"],
            "memory_usage": stats["peak_mem"],
            "avg_cpu": stats["avg_cpu"],
            "avg_mem": stats["avg_mem"]
        }
        return result, metrics

    @abc.abstractmethod
    def translate(self, text: str):
        pass
