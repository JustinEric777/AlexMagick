import gradio as gr
from pages.audios import asr, tts


def reload_default_model():
    print(22222)
    asr.asr.reload_model(default=True)


def create_ui(params: dict):
    with gr.Tab("Audio Models", id="audio_tab") as audio_tab:
        select_tab = params["default_second_tab"] if params["default_first_tab"] == audio_tab else None
        with gr.Tabs(selected=select_tab):
            asr.create_ui(params)
            tts.create_ui(params)

    audio_tab.select(reload_default_model)
