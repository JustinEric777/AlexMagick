import torch
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import load_image
from modules.models.images.text2img.base_model import BaseModel


class ModelStableDiffusion(BaseModel):
    def load_model(self, model_path: str, device: str):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        pipline = StableDiffusionInpaintPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=True,
        ).to(device)

        self.pipline = pipline
        self.device = device

    def generate(self, image, mask_image, positive_prompt, negative_prompt, seed, guidance_scale, num_inference_steps, height, width):
        generator = torch.Generator(self.device).manual_seed(seed)
        init_image = load_image(image)
        mask_image = load_image(mask_image)

        return self.pipline(
            image=init_image,
            mask_image=mask_image,
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator,
        ).images[0]

    def release(self):
        del self.pipline



