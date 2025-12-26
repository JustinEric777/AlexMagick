from typing import Dict, Any

TASK_TYPE = "sequence-llm"

MODEL_LIST: Dict[str, Any] = {
    "Pytorch": {
        "Qwen": {
            "model_provider_path": "core.models.sequences.llm.model_qwen_transformer",
            "model_provider_name": "QwenTransformerModel",
            "model_path": "/data/models/llm/pytorch/qwen3",
            "model_list": [
                "Qwen3-14B",
                "Qwen3-1.7B",
                "Qwen3-4B",
                "Qwen3-8B"
            ]
        }
    },
    "llama.cpp": {
        "DeepSeek": {
            "model_provider_path": "core.models.sequences.llm.model_deepseek_llama_cpp",
            "model_provider_name": "DeepSeekLlamaCppModel",
            "model_path": "/data/models/llm/llama.cpp/deepseek",
            "model_list": [
                "DeepSeek-R1-Distill-Qwen-7B-GGUF/DeepSeek-R1-Distill-Qwen-7B-Q8_0.gguf"
            ]
        },
    },
    "OpenVino": {
        "DeepSeek": {
            "model_provider_path": "core.models.sequences.llm.model_deepseek_openvino",
            "model_provider_name": "DeepSeekOpenvinoModel",
            "model_path": "/data/models/llm/openvino/deepseek",
            "model_list": [
                "DeepSeek-R1-Distill-Qwen-7B-openvino-int8",
            ]
        },
    },
    "ONNX": {
        "llama3": {
            "model_provider_path": "core.models.sequences.llm.model_llama_onnxruntime",
            "model_provider_name": "LlamaOnnxRunTimeModel",
            "model_path": "/data/models/llm/onnx/llama3",
            "model_list": [
                "Meta-Llama-3.1-8B-Instruct-onnx-fp16",
            ]
        },
    }
}
