from typing import Any

class BackendLoader:
    """
    Factory class to lazily load and instantiate inference backends.
    """
    @staticmethod
    def get_backend(backend_name: str, device: str, **kwargs) -> Any:
        backend_name = (backend_name or "").lower()

        if backend_name in ["transformer", "pytorch"]:
            from core.models.backends.torch import TorchBackend
            return TorchBackend(device=device)
            
        elif backend_name == "llama_cpp":
            from core.models.backends.llamacpp import LlamaCppBackend
            return LlamaCppBackend(device=device)
            
        elif backend_name == "openvino":
            from core.models.backends.openvino import OpenVINOBackend
            return OpenVINOBackend(device=device)
            
        elif backend_name in ["onnxruntime", "onnx"]:
            from core.models.backends.onnx import ONNXRuntimeBackend
            return ONNXRuntimeBackend(device=device)

        elif backend_name in ["ipex_llm", "ipex"]:
             from core.models.backends.ipexllm import IpexLLMBackend
             return IpexLLMBackend(device=device)

        else:
            raise ValueError(f"Unsupported backend: {backend_name}")

    @staticmethod
    def load(backend_name: str, device: str, model_path: str, **kwargs) -> Any:
        """
        Instantiate backend, configure default parameters, and load the model.
        """
        backend_name = (backend_name or "").lower()
        
        # 1. Apply Backend-Specific Defaults
        if backend_name in ["transformer", "pytorch"]:
            import torch
            # Common defaults for Transformer/Pytorch
            if "dtype" not in kwargs:
                kwargs["dtype"] = torch.bfloat16
            
            kwargs.setdefault("trust_remote_code", True)
            kwargs.setdefault("low_cpu_mem_usage", True)
            
            if device != "cpu":
                kwargs.setdefault("device_map", device.lower())
            else:
                kwargs.setdefault("device_map", "cpu")

        elif backend_name == "llama_cpp":
            kwargs.setdefault("n_ctx", 2048)

        elif backend_name == "openvino":
            kwargs.setdefault("version", "opset8")

        elif backend_name in ["onnxruntime", "onnx"]:
            kwargs.setdefault("trust_remote_code", True)
            if "dtype" not in kwargs:
                import torch
                kwargs["dtype"] = torch.bfloat16

        # 2. Get Backend Instance
        backend = BackendLoader.get_backend(backend_name, device)

        # 3. Load Model
        # Note: Some backends might use different loading methods, 
        # but current standardization implies load_text_generation for LLMs.
        # If this Loader is generic for other tasks, we might need to know the task type.
        # For now, assuming this is used by LLMs as per context.
        if hasattr(backend, "load_text_generation"):
            backend.load_text_generation(model_path, **kwargs)
        else:
            # Fallback or generic load if available
            if hasattr(backend, "load"):
                backend.load(model_path, **kwargs)
        
        return backend
