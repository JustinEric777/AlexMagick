from typing import Dict, Any

TASK_TYPE = "image-text2img"

MODEL_LIST: Dict[str, Any] = {
    "Pytorch": {
        "SD-1.5": {
            "model_provider_path": "modules.models.images.text2img.model_stable_diffusion",
            "model_provider_name": "ModelStableDiffusion",
            "model_path": "/data/models/image_gen",
            "model_list": [
                "stable-diffusion-v1-5"
            ]
        },
        "SD-2": {
            "model_provider_path": "modules.models.images.text2img.model_stable_diffusion_2",
            "model_provider_name": "ModelStableDiffusion2",
            "model_path": "/data/models/image_gen",
            "model_list": [
                "stable-diffusion-2-1"
            ]
        },
        "SD-XL": {
            "model_provider_path": "modules.models.images.text2img.model_stable_diffusion_xl",
            "model_provider_name": "ModelStableDiffusionXL",
            "model_path": "/data/models/image_gen",
            "model_list": [
                "sdxl-turbo"
            ]
        },
        "SD-3": {
            "model_provider_path": "modules.models.images.text2img.model_stable_diffusion_3",
            "model_provider_name": "ModelStableDiffusion3",
            "model_path": "/data/models/image_gen",
            "model_list": [
                "stable-diffusion-3-medium-diffusers"
            ]
        }
    }

}
