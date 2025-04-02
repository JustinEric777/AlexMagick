from config.text2img_config import MODEL_LIST, TASK_TYPE
from modules.servers.base_server import BaseServer, Metric


class Text2ImageServer(BaseServer):
    def __init__(self):
        super().__init__()
        self.task_type = TASK_TYPE
        self.model_list = MODEL_LIST

    @Metric()
    def generate(self, positive_prompt, negative_prompt, seed, guidance_scale, num_inference_steps, width, height):
        return self.pipeline.generate(
            positive_prompt,
            negative_prompt,
            seed,
            guidance_scale,
            num_inference_steps,
            width,
            height
        )
