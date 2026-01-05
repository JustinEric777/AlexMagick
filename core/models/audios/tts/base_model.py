import abc
import os
import sys
from typing import Any
from core.models.loader import BaseModel as LoaderBaseModel

from core.monitor.system import SystemMonitor
from typing import Any, Tuple, Dict

sub_module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
if sub_module_path not in sys.path:
    sys.path.insert(0, sub_module_path)

AUDIO_PATH = "storage/audio/tts"


class BaseModel(LoaderBaseModel):
    vocoder: Any = None
    dtype: Any = None

    def generate(self, texts: [str], sample_wav: str = None) -> Tuple[str, Dict[str, Any]]:
        tracker = SystemMonitor.Tracker()
        with tracker:
            result = self.inference(texts, sample_wav)
        
        stats = tracker.get_stats()
        metrics = {
            "cpu_usage": stats["peak_cpu"],
            "memory_usage": stats["peak_mem"],
            "avg_cpu": stats["avg_cpu"],
            "avg_mem": stats["avg_mem"]
        }
        return result, metrics

    @abc.abstractmethod
    def inference(self, text: [str], sample_wav: str):
        pass
