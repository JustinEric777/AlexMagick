import gradio as gr
from pages.images import stable_diffusion


def create_ui(params: dict):
    with gr.Tab("Image Models", id="image_tab") as image_tab:
        with gr.Tabs(selected=params["default_second_tab"]):
            stable_diffusion.create_ui(params)

    image_tab.select(stable_diffusion.reload_model)
