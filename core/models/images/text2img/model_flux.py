import torch
from diffusers import FluxPipeline
from core.models.images.text2img.base_model import BaseModel

class ModelFlux(BaseModel):
    def load_model(self, model_path: str, device: str):
        pipe = FluxPipeline.from_pretrained(
            model_path, 
            dtype=torch.bfloat16
        )
        if device != "cpu":
            pipe.enable_model_cpu_offload()
        
        self.pipeline = pipe
        self.device = device

    def _generate(self, positive_prompt: str, negative_prompt: str, seed: int = 0, guidance_scale: float = 3.5, num_inference_steps: int = 20, width: int = 1024, height: int = 1024):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        image = self.pipeline(
            prompt=positive_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=512,
            generator=generator
        ).images[0]
        
        return image

    def release(self):
        del self.pipeline
