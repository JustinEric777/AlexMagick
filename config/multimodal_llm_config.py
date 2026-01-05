from typing import Dict, Any

TASK_TYPE = "multimodal_llm"

MODEL_LIST: Dict[str, Any] = {
    "Pytorch": {
        "Qwen2.5-Omni": {
            "model_provider_path": "core.models.multimodals.llm.model_qwen_omni_transformer",
            "model_provider_name": "QwenOmiTransformerModel",
            "model_path": "/data/models/llm/pytorch/qwen-omni",
            "model_list": [
                "Qwen2.5-Omni-3B",
                "Qwen2.5-Omni-7B",
                "Qwen3-Omni-30B-A3B-Instruct"
            ]
        },
        "Qwen2.5-Omni-GPTQ-4bit": {
            "model_provider_path": "core.models.multimodals.llm.model_qwen_omni_transformer_gptq",
            "model_provider_name": "QwenOmiTransformerGPTQModel",
            "model_path": "/data/models/llm/pytorch/qwen-omni",
            "model_list": [
                # "Qwen2.5-Omni-7B-GPTQ-4bit"
            ]
        },
        "Qwen-VL": {
            "model_provider_path": "core.models.multimodals.llm.model_qwen_vl",
            "model_provider_name": "QwenVLModel",
            "model_path": "/data/models/llm/pytorch/qwen-vl",
            "model_list": [
                "Qwen3-VL-2B-Instruct",
                "Qwen3-VL-4B-Instruct"
            ]
        },
        "MiniCPM-V": {
            "model_provider_path": "core.models.multimodals.llm.model_minicpm_v",
            "model_provider_name": "MiniCPMVModel",
            "model_path": "/data/models/llm/pytorch",
            "model_list": [
                "MiniCPM-V-4_5",
                "minicpm/MiniCPM-V-2_6"
            ]
        }
    },
    "llama.cpp": {
        "Qwen2.5-Omni": {
            "model_provider_path": "core.models.multimodals.llm.model_qwen_omni_llama_cpp",
            "model_provider_name": "QwenOmiLLamaCppModel",
            "model_path": "/data/models/llm/llama.cpp/Qwen2.5-Omni",
            "model_list": [
                "Qwen2.5-Omni-3B-GGUF/Qwen2.5-Omni-3B-f16.gguf",
                # "Qwen2.5-Omni-3B-GGUF/Qwen2.5-Omni-3B-Q8_0.gguf",
                # "Qwen2.5-Omni-3B-GGUF/Qwen2.5-Omni-3B-Q4_K_M.gguf",
                # "Qwen2.5-Omni-7B-GGUF/Qwen2.5-Omni-7B-f16.gguf",
                # "Qwen2.5-Omni-7B-GGUF/Qwen2.5-Omni-7B-Q8_0.gguf",
                # "Qwen2.5-Omni-7B-GGUF/Qwen2.5-Omni-7B-Q4_K_M.gguf",
            ]
        },
    }
}
