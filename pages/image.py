import gradio as gr
from pages.images import stable_diffusion


def create_ui(params: dict):
    with gr.Tab("Image Models", id="image_tab"):
        with gr.Tabs(selected=params["default_second_tab"]):
            with gr.Tab("Stable Diffusion", id="sd_tab"):
                stable_diffusion.create_ui(params)
