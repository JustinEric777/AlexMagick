import logging
from config.mt_config import MODEL_LIST, TASK_TYPE
from servers.base_server import BaseServer

logger = logging.getLogger(__name__)

class MtServer(BaseServer):
    TASK_TYPE = TASK_TYPE
    MODEL_LIST = MODEL_LIST

    def __init__(self):
        super().__init__()

    def generate(self, text: str) -> (str, str):
        def _impl():
            start_time = self.pre_generate()
            output, model_metrics = self.pipeline.generate(text)
            
            extra = {
                "words_count": len(output),
                **model_metrics
            }
            metric = self.post_generate(start_time, extra)
            return output, metric
            
        return self.safe_run(_impl, default=("", "Error occurred"))
