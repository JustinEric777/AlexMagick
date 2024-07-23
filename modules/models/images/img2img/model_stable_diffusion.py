import torch
from diffusers import StableDiffusionImg2ImgPipeline
from modules.models.images.text2img.base_model import BaseModel


class ModelStableDiffusion(BaseModel):
    def load_model(self, model_path: str):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        pipline = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=True,
            use_safetensors=True,
            low_cpu_mem_usage=True,
        ).to(device)

        self.pipline = pipline
        self.device = device

    def generate(self, image, positive_prompt, negative_prompt, seed, guidance_scale, num_inference_steps, height, width):
        generator = torch.Generator(self.device).manual_seed(seed)
        image = self.pipline(
            image=image,
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator,
        ).image[0]

        return image



