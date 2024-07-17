from typing import Dict, Any

MODEL_LIST: Dict[str, Any] = {
    "llama3": {
        "Transformers": {
            "model_provider_path": "models.sequences.llm.model_llama_transformer",
            "model_provider_name": "LlamaTransformerModel",
            "model_path": "/data/models/llm/llama3/torch",
            "model_list": [
                "Meta-Llama-3-8B",
                "Meta-Llama-3-8B-Instruct",
                "Llama3-8B-Chinese-Chat",
                "Unichat-llama3-Chinese-8B"
            ]
        },
        "llama.cpp": {
            "model_provider_path": "models.sequences.llm.model_llama_cpp",
            "model_provider_name": "LlamaCppModel",
            "model_path": "/data/models/llm/llama3/gguf",
            "model_list": [
                "Meta-Llama-3-8B-Instruct.gguf",
                "Meta-Llama-3-8B-Instruct-Q8_0.gguf",
                "Meta-Llama-3-8B-Instruct-Q6_K.gguf",
                "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
            ]
        },
        "OpenVino": {
            "model_provider_path": "models.sequences.llm.model_llama_openvino",
            "model_provider_name": "LlamaOpenvinoModel",
            "model_path": "/data/models/llm/llama3/torch",
            "model_list": [
                "Meta-Llama-3-8B",
                "Meta-Llama-3-8B-Instruct"
            ]
        },
        "IPEX-LLM": {
            "model_provider_path": "models.sequences.llm.model_llama_ipexllm",
            "model_provider_name": "LlamaIpexLLMModel",
            "model_path": "/data/models/llm/llama3/torch",
            "model_list": [
                "Meta-Llama-3-8B",
                "Meta-Llama-3-8B-Instruct"
            ]
        },
    },
    "glm_4v": {
        "Transformers": {
            "model_provider_path": "models.llm.model_glm4v_transformer",
            "model_provider_name": "LlamaTransformerModel",
            "model_path": "/data/models/llm/glm-4v/pytorch",
            "model_list": [
                "glm-4v-9b",
                "glm-4v-9b-chat",
            ]
        }
    }
}