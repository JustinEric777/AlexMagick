# Alex Magick Introduction & User Guide

## Introduction

Alex Magick is a comprehensive AI platform that unifies the capabilities of modern artificial intelligence into a single, accessible workspace. It is designed for developers, researchers, and enthusiasts who want to experiment with and utilize various AI models without the hassle of setting up separate environments for each.

The core philosophy of Alex Magick is **Unification and Efficiency**. It abstracts away the complexities of different inference backends (like PyTorch and OpenVINO) and model architectures, providing a consistent interface for interaction. Whether you are generating art with Stable Diffusion, chatting with a Large Language Model, or synthesizing speech, the experience is seamless.

### Architecture Highlights

-   **Lazy Loading & Singleton Engine**: To optimize resource usage, especially on consumer GPUs, Alex Magick ensures that only the active model occupies memory. Switching between tasks (e.g., from Image Generation to Chat) automatically unloads the previous model.
-   **Adapter Pattern**: The system uses a sophisticated adapter layer (`core.models.adapters`) to standardize calls. This means adding a new model often just requires a configuration update and a simple wrapper, rather than a full rewrite.
-   **Vector Database Support**: Integrated **Milvus Lite** for local, standalone vector storage (no Docker required). Data is persisted in `db/milvus/data`.

## User Guide

### 1. Getting Started

After installation (see `README.md`), launch the application:
```bash
python webui.py
```
Open your browser and navigate to `http://localhost:7860`.

### 2. Interface Overview

The interface is divided into main tabs representing different AI domains:

-   **Sequence**: Text-based tasks (LLM Chat, Translation, Embeddings).
-   **Image**: Visual generation and editing (Text-to-Image, Image-to-Image, Inpainting).
-   **Audio**: Speech processing (Text-to-Speech, Automatic Speech Recognition).
-   **Video**: Video analysis (Embedding).
-   **Multimodal**: Models that handle multiple input types (e.g., Image+Text chat).

### 3. Task Walkthroughs

#### A. Chatting with an LLM (Sequence -> LLM Model)
1.  Navigate to the **Sequence** tab, then select **LLM Model**.
2.  On the right sidebar, use "Reload Model" to select your desired Architecture (e.g., Pytorch) and Model (e.g., Qwen).
3.  Adjust parameters like `temperature` (creativity) and `max_tokens` (length).
4.  Type your message in the chat box and press **Send**.

#### B. Generating Images (Image -> Text2Image Model)
1.  Navigate to the **Image** tab, then **Text2Image Model**.
2.  Enter a **Positive Prompt** describing what you want to see.
3.  (Optional) Enter a **Negative Prompt** for what to avoid.
4.  Adjust `Width`, `Height`, and `Inference Steps` on the right.
5.  Click **Generate**. The image will appear in the output panel.

#### C. Text-to-Speech (Audio -> TTS Model)
1.  Navigate to the **Audio** tab, then **TTS Model**.
2.  Select a model like `ChatTTS` or `CosyVoice` from the right panel.
3.  Enter the text you want to convert into speech.
4.  Click **Generate**. You can listen to or download the resulting audio file.

### 4. Advanced Usage

-   **Model Switching**: You can switch models on the fly using the right-hand configuration panel in each tab. The system will handle the loading/unloading.
-   **History**: The LLM tab supports viewing chat history and metadata.
-   **API Access**: Since the UI is built on Gradio, you can also access the functionalities via API endpoints (check the "Use via API" link at the bottom of the Gradio page if enabled).

## Troubleshooting

-   **OOM (Out of Memory)**: If you encounter memory errors, ensure no other heavy applications are running. The system automatically frees memory when switching tabs, but running very large models (like SDXL or 70B LLMs) still requires sufficient hardware.
-   **Model Not Found**: Check if the model weights are correctly placed in the `data/models/` directory as specified in the `config/*.py` files.
