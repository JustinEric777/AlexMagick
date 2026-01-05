from typing import Dict, Any

TASK_TYPE = "llm"

MODEL_LIST: Dict[str, Any] = {
    "Pytorch": {
        "Qwen": {
            "model_provider_path": "core.models.sequences.llm.model_qwen",
            "model_provider_name": "QwenModel",
            "model_path": "/data/models/llm/pytorch/qwen3",
            "model_list": [
                "Qwen3-14B",
                "Qwen3-1.7B",
                "Qwen3-4B",
                "Qwen3-8B"
            ]
        },
        "GPT2": {
            "model_provider_path": "core.models.sequences.llm.model_gpt2",
            "model_provider_name": "GPT2Model",
            "model_path": "/data/models/llm/pytorch/gpt",
            "model_list": [
                "gpt2"
            ]
        },
        "GPT-OSS": {
            "model_provider_path": "core.models.sequences.llm.model_gpt_oss",
            "model_provider_name": "GPTOSSModel",
            "model_path": "/data/models/llm/pytorch",
            "model_list": [
                "gpt-oss-20b"
            ]
        },
        "Qwen3-Thinking": {
             "model_provider_path": "core.models.sequences.llm.model_qwen",
             "model_provider_name": "QwenModel",
             "model_path": "/data/models/llm/pytorch",
             "model_list": [
                 "Qwen3-Next-80B-A3B-Thinking"
             ]
        }
    },
    "llama.cpp": {
        "DeepSeek": {
            "model_provider_path": "core.models.sequences.llm.model_deepseek",
            "model_provider_name": "DeepSeekModel",
            "model_path": "/data/models/llm/llama.cpp/deepseek",
            "model_list": [
                "DeepSeek-R1-Distill-Qwen-7B-GGUF/DeepSeek-R1-Distill-Qwen-7B-Q8_0.gguf",
                "DeepSeek-R1-Distill-Qwen-14B-GGUF/DeepSeek-R1-Distill-Qwen-14B-Q8_0.gguf"
            ]
        },
        "Qwen": {
            "model_provider_path": "core.models.sequences.llm.model_qwen",
            "model_provider_name": "QwenModel",
            "model_path": "/data/models/llm/llama.cpp/Qwen",
            "model_list": [
                "Qwen_QwQ-32B-GGUF/Qwen_QwQ-32B-Q4_K_M.gguf"
            ]
        }
    },
    "OpenVino": {
        "DeepSeek": {
            "model_provider_path": "core.models.sequences.llm.model_deepseek",
            "model_provider_name": "DeepSeekModel",
            "model_path": "/data/models/llm/openvino/deepseek",
            "model_list": [
                "DeepSeek-R1-Distill-Qwen-7B-openvino-int8",
            ]
        },
    },
    "ONNX": {
        "llama3": {
            "model_provider_path": "core.models.sequences.llm.model_llama",
            "model_provider_name": "LlamaModel",
            "model_path": "/data/models/llm/onnx/llama3",
            "model_list": [
                "Meta-Llama-3.1-8B-Instruct-onnx-fp16",
            ]
        },
    }
}
