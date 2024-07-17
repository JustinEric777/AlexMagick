import torch
from diffusers import StableDiffusion3Pipeline
from transformers import T5EncoderModel, BitsAndBytesConfig
from modules.models.images.image_gen.base_model import BaseModel


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

        model = StableDiffusion3Pipeline.from_pretrained(
            model_path,
            text_encoder_3=text_encoder,
            device_map="balanced",
            torch_dtype=dtype,
            device=device
        )

        self.model = model

    def generate(self, save_image_name, prompt, negative_prompt, num_inference_steps, height, width, guidance_scale):
        image = self.model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance_scale=guidance_scale
        ).images[0]
        image.save(save_image_name)

        return save_image_name



