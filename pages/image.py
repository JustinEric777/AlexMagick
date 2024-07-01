import gradio as gr
from pages.images import stable_diffusion


def create_ui(params: dict):
    with gr.Tab("Image Models", id="image_tabs"):
        with gr.Tab("Stable Diffusion", id="image_sd_tab"):
            stable_diffusion.create_ui(params)