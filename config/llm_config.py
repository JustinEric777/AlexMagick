from typing import Dict, Any

TASK_TYPE = "sequence-llm"

MODEL_LIST: Dict[str, Any] = {
    "Pytorch": {
        "llama3": {
            "model_provider_path": "modules.models.sequences.llm.model_llama_transformer",
            "model_provider_name": "LlamaTransformerModel",
            "model_path": "/data/models/llm/pytorch/llama3",
            "model_list": [
                "Meta-Llama-3.1-8B-Instruct",
                "Meta-Llama-3-8B-Instruct",
                "Llama3-8B-Chinese-Chat",
            ]
        },
        "glm4": {
            "model_provider_path": "modules.models.sequences.llm.model_glm4v_transformer",
            "model_provider_name": "Glm4vTransformerModel",
            "model_path": "/data/models/llm/pytorch/glm4",
            "model_list": [
                "glm-4-9b-chat",
                "glm-4-9b",
            ]
        },
        "minicpm": {
            "model_provider_path": "modules.models.sequences.llm.model_minicpm_transformer",
            "model_provider_name": "MiniCPMTransformerModel",
            "model_path": "/data/models/llm/pytorch/minicpm",
            "model_list": [
                "MiniCPM-V-2_6"
            ]
        }
    },
    "llama.cpp": {
        "llama3": {
            "model_provider_path": "modules.models.sequences.llm.model_llama_cpp",
            "model_provider_name": "LlamaCppModel",
            "model_path": "/data/models/llm/llama.cpp/llama3",
            "model_list": [
                "Meta-Llama-3-8B-Instruct-BF16.gguf",
                "Meta-Llama-3-8B-Instruct-Q8_0.gguf",
                "Meta-Llama-3.1-8B-Instruct-BF16.gguf",
                "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
            ]
        }
    },
    "OpenVino": {
        "llama3": {
            "model_provider_path": "modules.models.sequences.llm.model_llama_openvino",
            "model_provider_name": "LlamaOpenvinoModel",
            "model_path": "/data/models/llm/openvino/llama3",
            "model_list": [
                "Meta-Llama-3.1-8B-Instruct-Openvino-fp16",
            ]
        }
    },
    "ONNX": {
        "llama3": {
            "model_provider_path": "modules.models.sequences.llm.model_llama_onnxruntime",
            "model_provider_name": "LlamaOnnxRunTimeModel",
            "model_path": "/data/models/llm/onnx/llama3",
            "model_list": [
                "Meta-Llama-3.1-8B-Instruct-onnx-fp16",
            ]
        },
    }
}
