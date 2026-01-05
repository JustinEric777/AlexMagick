import torch
from diffusers import HunyuanDiTPipeline
from core.models.images.text2img.base_model import BaseModel

class ModelHunyuanDiT(BaseModel):
    def load_model(self, model_path: str, device: str):
        pipe = HunyuanDiTPipeline.from_pretrained(
            model_path, 
            dtype=torch.float16
        )
        if device != "cpu":
            pipe.to(device)
        
        self.pipeline = pipe
        self.device = device

    def _generate(self, positive_prompt: str, negative_prompt: str, seed: int = 0, guidance_scale: float = 5.0, num_inference_steps: int = 20, width: int = 1024, height: int = 1024):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        image = self.pipeline(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
        ).images[0]
        
        return image

    def release(self):
        del self.pipeline
