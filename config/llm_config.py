from typing import Dict, Any

TASK_TYPE = "sequence-llm"

MODEL_LIST: Dict[str, Any] = {
    "Pytorch": {
        "DeepSeek": {
            "model_provider_path": "modules.models.sequences.llm.model_deepseek_transformer",
            "model_provider_name": "DeepSeekTransformerModel",
            "model_path": "/data/models/llm/pytorch/deepseek",
            "model_list": [
                "DeepSeek-R1-Distill-Llama-8B",
                "DeepSeek-R1-Distill-Qwen-7B",
                "DeepSeek-R1-Distill-Qwen-1.5B",
            ]
        },
        "llama3": {
            "model_provider_path": "modules.models.sequences.llm.model_llama_transformer",
            "model_provider_name": "LlamaTransformerModel",
            "model_path": "/data/models/llm/pytorch/llama3",
            "model_list": [
                "Meta-Llama-3-8B-Instruct",
                "Meta-Llama-3.1-8B-Instruct",
                "Llama-3.2-1B-Instruct",
                "Llama-3.2-3B-Instruct",
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
                "MiniCPM-V-2_6",
            ]
        },
        "minicpm-o-2.6": {
            "model_provider_path": "modules.models.sequences.llm.model_minicpm_o_transformer",
            "model_provider_name": "MiniCPMOTransformerModel",
            "model_path": "/data/models/llm/pytorch/minicpm",
            "model_list": [
                "MiniCPM-o-2_6",
            ]
        },
        "Qwen": {
            "model_provider_path": "modules.models.sequences.llm.model_qwen_transformer",
            "model_provider_name": "QwenTransformerModel",
            "model_path": "/data/models/llm/pytorch/qwen",
            "model_list": [
                "Qwen2.5-1.5B-Instruct",
                "Qwen2.5-3B-Instruct",
            ]
        }
    },
    "llama.cpp": {
        "DeepSeek": {
            "model_provider_path": "modules.models.sequences.llm.model_deepseek_llama_cpp",
            "model_provider_name": "DeepSeekLlamaCppModel",
            "model_path": "/data/models/llm/llama.cpp/deepseek",
            "model_list": [
                "DeepSeek-R1-Distill-Qwen-14B-GGUF/DeepSeek-R1-Distill-Qwen-14B-Q8_0.gguf",
                "DeepSeek-R1-Distill-Qwen-14B-GGUF/DeepSeek-R1-Distill-Qwen-14B-Q4_1.gguf",
                "DeepSeek-R1-Distill-Qwen-7B-GGUF/DeepSeek-R1-Distill-Qwen-7B-f16.gguf",
                "DeepSeek-R1-Distill-Qwen-7B-GGUF/DeepSeek-R1-Distill-Qwen-7B-Q8_0.gguf",
                "DeepSeek-R1-Distill-Qwen-7B-GGUF/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
                "DeepSeek-R1-Distill-Qwen-1.5B-GGUF/DeepSeek-R1-Distill-Qwen-1.5B-f16.gguf",
                "DeepSeek-R1-Distill-Qwen-1.5B-GGUF/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf",
                "DeepSeek-R1-Distill-Qwen-1.5B-GGUF/DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf",
            ]
        },
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
        },
    },
    "OpenVino": {
        "deepseek": {
            "model_provider_path": "modules.models.sequences.llm.model_deepseek_openvino",
            "model_provider_name": "DeepSeekOpenvinoModel",
            "model_path": "/data/models/llm/openvino/deepseek",
            "model_list": [
                "DeepSeek-R1-Distill-Llama-8B-openvino-int8",
                "DeepSeek-R1-Distill-Llama-8B-openvino-int4",
                "DeepSeek-R1-Distill-Qwen-7B-openvino-int8",
                "DeepSeek-R1-Distill-Qwen-7B-openvino-int4",
                "DeepSeek-R1-Distill-Qwen-1.5B-openvino-int8",
                # "DeepSeek-R1-Distill-Llama-8B_GPTQ-openvino-int4",
                # "DeepSeek-R1-Distill-Qwen-7B-sym-int4",
            ]
        },
        "llama3": {
            "model_provider_path": "modules.models.sequences.llm.model_llama_openvino",
            "model_provider_name": "LlamaOpenvinoModel",
            "model_path": "/data/models/llm/openvino/llama3",
            "model_list": [
                # "Meta-Llama-3.1-8B-Instruct-Openvino-fp16",
                "llama-3-8b-instruct-ov-int8",
            ]
        },
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
