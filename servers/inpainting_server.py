from config.inpainting_config import MODEL_LIST, TASK_TYPE
from servers.base_server import BaseImageGenServer
from core.monitor import record_task


class InpaintingServer(BaseImageGenServer):
    TASK_TYPE = TASK_TYPE
    MODEL_LIST = MODEL_LIST

    def __init__(self):
        super().__init__()

    @record_task("Inpainting", "Image Inpainting")
    def generate(self, image, mask_image, positive_prompt, negative_prompt, seed, guidance_scale, num_inference_steps, width, height):
        return self.run_image_generation(
            image,
            mask_image,
            positive_prompt,
            negative_prompt,
            seed,
            guidance_scale,
            num_inference_steps,
            width,
            height
        )
