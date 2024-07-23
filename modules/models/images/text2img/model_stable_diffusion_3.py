import torch
from diffusers import StableDiffusion3Pipeline
from transformers import T5EncoderModel, BitsAndBytesConfig
from modules.models.images.text2img.base_model import BaseModel


class ModelStableDiffusion3(BaseModel):
    def load_model(self, model_path: str):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        text_encoder = T5EncoderModel.from_pretrained(
            model_path,
            subfolder="text_encoder_3",
            quantization_config=quantization_config,
            device=device
        )

        pipline = StableDiffusion3Pipeline.from_pretrained(
            model_path,
            text_encoder_3=text_encoder,
            device_map="balanced",
            torch_dtype=dtype,
            device=device
        ).to(device)

        self.pipline = pipline
        self.device = device

    def generate(self, positive_prompt, negative_prompt, seed, guidance_scale, num_inference_steps, height, width):
        image = self.pipline(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height
        ).images[0]

        return image



