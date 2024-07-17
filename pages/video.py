import gradio as gr
from pages.audios import asr


def create_ui(params: dict):
    with gr.Tab("Video Models", id="video_tab") as video_tab:
        with gr.Tabs(selected=params["default_second_tab"]):
            asr.create_ui(params)

    video_tab.select(asr.reload_model)
