from typing import Dict, Any

TASK_TYPE = "image-img2img"

MODEL_LIST: Dict[str, Any] = {
    "Pytorch": {
        "SD-1.5": {
            "model_provider_path": "modules.models.images.img2img.model_stable_diffusion",
            "model_provider_name": "ModelStableDiffusion",
            "model_path": "/data/models/image_gen",
            "model_list": [
                "stable-diffusion-v1-5"
            ]
        },
        "SD-XL": {
            "model_provider_path": "modules.models.images.img2img.model_stable_diffusion_xl",
            "model_provider_name": "ModelStableDiffusionXL",
            "model_path": "/data/models/image_gen",
            "model_list": [
                "sdxl-turbo"
            ]
        }
    }
}
