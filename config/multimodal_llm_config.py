from typing import Dict, Any

TASK_TYPE = "multimodal-mllm"

MODEL_LIST: Dict[str, Any] = {
    "Pytorch": {
        "Qwen2.5-Omni": {
            "model_provider_path": "modules.models.multimodals.llm.model_qwen_omni_transformer",
            "model_provider_name": "QwenOmiTransformerModel",
            "model_path": "/data/models/llm/pytorch/qwen/",
            "model_list": [
                "Qwen2.5-Omni-7B"
            ]
        },
        "Qwen2.5-Omni-GPTQ-4bit": {
            "model_provider_path": "modules.models.multimodals.llm.model_qwen_omni_transformer_gptq",
            "model_provider_name": "QwenOmiTransformerGPTQModel",
            "model_path": "/data/models/llm/pytorch/qwen/",
            "model_list": [
                "Qwen2.5-Omni-7B-GPTQ-4bit"
            ]
        }
    },
    # "llama.cpp": {
    # }
}
