# Core Architecture Design

## Overview
The `core` module provides a unified interface for model inference across various domains (LLM, ASR, TTS, etc.) and backends (PyTorch, OpenVINO, ONNX, etc.).

## Key Components

### 1. Adapters (`core.models.adapters`)
The central component for model resolution and unification.
- **`resolve_model_config(domain, infer_arch, model_name)`**: Standardizes the mapping from configuration parameters to specific backend and implementation names.
- **`_UnifiedAdapter`**: A generic adapter that loads the specific model implementation based on the resolution result. It wraps the implementation to expose a consistent API (e.g., `generate`, `chat`).

### 2. Backends (`core.models.backends`)
Encapsulates low-level inference engines.
- **`BackendRunner`**: Abstract base class for text generation backends.
- **Implementations**: `OpenVINOBackend`, `LlamaCppBackend`, `ONNXRuntimeBackend`, `TorchBackend`.
- **Usage**: Model implementations (e.g., `DeepSeekOpenvinoModel`) delegate the heavy lifting of loading and inference to these backends, ensuring consistent behavior and easier switching.

### 3. Model Implementations (`core.models.*`)
Domain-specific model classes (e.g., `core.models.sequences.llm`).
- Inherit from domain-specific `BaseModel`.
- Can use `Backends` for inference or implement custom logic.
- adhere to strict interfaces defined in `base_model.py`.

### 4. Model Engine (`core.models.engine.ModelEngine`)
The high-level entry point for servers and applications.
- Initializes the `ModelContext`.
- Uses `ModelFactory` to create the appropriate adapter/model.
- Provides a unified `generate` method.

## Architecture Flow

1.  **Request**: Server receives a request with `infer_arch`, `model_name`, etc.
2.  **Resolution**: Server calls `resolve_model_config` to determine `backend` and `impl`.
3.  **Initialization**: Server initializes `ModelEngine` with resolved parameters.
4.  **Loading**: `ModelEngine` uses `ModelFactory` -> `_UnifiedAdapter` -> `_resolve_impl_path` -> Import and Instantiate specific Model Class.
5.  **Inference**: Request -> `ModelEngine.generate` -> `_UnifiedAdapter.generate` -> `_ExternalWrapper` -> Specific Model Implementation -> Backend (optional).

## Backward Compatibility
- The `BaseServer` class is maintained but refactored to delegate logic to `core` components.
- Existing config files (`config/*.py`) remain compatible, with `resolve_model_config` handling the translation to new internal paths.

## Extensibility
- **New Domain**: Add `BaseModel` in `core/models/<domain>`, register in `ModelFactory` (or `adapters.py`), update `resolve_model_config`.
- **New Backend**: Add implementation in `core/models/backends/`, update `resolve_model_config` if needed.
