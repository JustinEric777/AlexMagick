import gradio as gr
from pages.images import text2img, img2img, inpainting


def reload_default_model():
    text2img.text2img.reload_model(default=True)


def create_ui(params: dict):
    with gr.Tab("Image Models", id="image_tab") as image_tab:
        select_tab = params["default_second_tab"] if params["default_first_tab"] == image_tab else None
        with gr.Tabs(selected=select_tab):
            text2img.create_ui(params)
            img2img.create_ui(params)
            inpainting.create_ui(params)

    image_tab.select(reload_default_model)
