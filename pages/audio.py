import gradio as gr
from pages.audios import asr


def reload_default_model():
    asr.asr.reload_model(default=True)


def create_ui(params: dict):
    with gr.Tab("Audio Models", id="audio_tab") as audio_tab:
        with gr.Tabs(selected=params["default_second_tab"]):
            asr.create_ui(params)

    audio_tab.select(reload_default_model)
