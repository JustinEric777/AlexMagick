import time
import logging
from config.asr_config import MODEL_LIST, TASK_TYPE
from servers.base_server import BaseServer
from core.monitor import record_task

logger = logging.getLogger(__name__)

class ASRServer(BaseServer):
    TASK_TYPE = TASK_TYPE
    MODEL_LIST = MODEL_LIST

    def __init__(self):
        super().__init__()

    @record_task("ASR", "Speech Recognition")
    def generate(self, audio: str):
        def _impl():
            start_time = self.pre_generate()
            outputs, model_metrics = self.pipeline.generate(audio)
            
            extra = {
                "words_count": len(outputs),
                "single_word_cost_time": round((time.time() - start_time) / max(len(outputs), 1), 3),
                **model_metrics
            }
            metric = self.post_generate(start_time, extra)
            return outputs, metric
            
        return self.safe_run(_impl, default=([], "Error occurred"))
