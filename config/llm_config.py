from typing import Dict, Any

TASK_TYPE = "sequence-llm"

MODEL_LIST: Dict[str, Any] = {
    "Pytorch": {
        "llama3": {
            "model_provider_path": "modules.models.sequences.llm.model_llama_transformer",
            "model_provider_name": "LlamaTransformerModel",
            "model_path": "/data/models/llm/pytorch/llama3",
            "model_list": [
                "Meta-Llama-3-8B-Instruct",
                "Meta-Llama-3-8B",
                "Llama3-8B-Chinese-Chat",
                "Unichat-llama3-Chinese-8B"
            ]
        },
        "llama3.1": {
            "model_provider_path": "modules.models.sequences.llm.model_llama_transformer",
            "model_provider_name": "LlamaTransformerModel",
            "model_path": "/data/models/llm/pytorch/llama3.1",
            "model_list": [
                "Meta-Llama-3.1-8B-Instruct",
            ]
        },
        "glm4": {
            "model_provider_path": "modules.models.llm.model_glm4v_transformer",
            "model_provider_name": "LlamaTransformerModel",
            "model_path": "/data/models/llm/pytorch/glm4",
            "model_list": [
                "glm-4v-9b",
                "glm-4v-9b-chat",
            ]
        }
    },
    "llama.cpp": {
        "llama3": {
            "model_provider_path": "modules.models.sequences.llm.model_llama_cpp",
            "model_provider_name": "LlamaCppModel",
            "model_path": "/data/models/llm/llama.cpp/llama3",
            "model_list": [
                "Meta-Llama-3-8B-Instruct.gguf",
                "Meta-Llama-3-8B-Instruct-Q8_0.gguf",
                "Meta-Llama-3-8B-Instruct-Q6_K.gguf",
                "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
            ]
        }
    },
    "OpenVino": {
        "llama3": {
            "model_provider_path": "modules.models.llm.model_llama_openvino",
            "model_provider_name": "LlamaOpenvinoModel",
            "model_path": "/data/models/llm/openvino/llama3",
            "model_list": [
                "Meta-Llama-3-8B",
                "Meta-Llama-3-8B-Instruct"
            ]
        }
    },
    "IPEX-LLM": {
        "llama3": {
            "model_provider_path": "modules.models.sequences.llm.model_llama_ipexllm",
            "model_provider_name": "LlamaIpexLLMModel",
            "model_path": "/data/models/llm/IPEX-LLM/llama3",
            "model_list": [
                "Meta-Llama-3-8B",
                "Meta-Llama-3-8B-Instruct"
            ]
        },
    }
}
