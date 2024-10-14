import gradio as gr
from pages.videos import embedding
from modules import video2embedding


def reload_default_model():
    video2embedding.reload_model(default=True)


def create_ui(params: dict):
    with gr.Tab("Video Models", id="video_tab") as video_tab:
        with gr.Tabs(selected=params["default_second_tab"]):
            embedding.create_ui(params)

    video_tab.select(reload_default_model)
