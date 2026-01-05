import time
import psutil
import torchaudio
import logging
from typing import List

from config.tts_config import MODEL_LIST, TASK_TYPE
from servers.base_server import BaseServer
from core.monitor import record_task

logger = logging.getLogger(__name__)

class TTSServer(BaseServer):
    TASK_TYPE = TASK_TYPE
    MODEL_LIST = MODEL_LIST

    def __init__(self):
        super().__init__()

    @record_task("TTS", "Text to Speech")
    def generate(self, texts: List[str]):
        def _impl():
            start_time = self.pre_generate()
            process = psutil.Process()
            
            # Run
            outputs, model_metrics = self.pipeline.generate(texts)
            
            # Metrics
            rtf = "N/A"
            tokens_per_second = "N/A"
            if outputs and isinstance(outputs, str):
                 try:
                     info = torchaudio.info(outputs)
                     duration = info.num_frames / info.sample_rate
                     cost = time.time() - start_time
                     if duration > 0:
                         rtf = round(cost / duration, 3)
                     
                     total_chars = sum(len(t) for t in texts)
                     if cost > 0:
                         tokens_per_second = round(total_chars / cost, 2)
                 except: pass
            
            extra = {
                "text_length": len(texts),
                "rtf": rtf,
                "tokens_per_second": tokens_per_second,
                **model_metrics
            }
            metric = self.post_generate(start_time, extra)
            return outputs, metric

        return self.safe_run(_impl, default=(None, "Error occurred"))
