import gradio as gr
from pages.audios import asr


def create_ui(params: dict):
    with gr.Tab("Video Models", id="audio_tabs"):
        with gr.Tab("ASR Model", id="audio_asr_tab"):
            asr.create_ui(params)