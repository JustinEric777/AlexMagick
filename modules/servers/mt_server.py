from config.mt_config import MODEL_LIST, TASK_TYPE
from modules.servers.base_server import BaseServer, Metric


class MtServer(BaseServer):
    def __init__(self):
        super().__init__()
        self.task_type = TASK_TYPE
        self.model_list = MODEL_LIST

    @Metric()
    def generate(self, text: str, model_name: str) -> (str, str):
        return self.pipeline_object.translate(text)



