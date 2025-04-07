import gradio as gr
from pages.multimodals import multimodal_llm


def reload_default_model():
    multimodal_llm.multimodal_llm.reload_model(default=True)


def create_ui(params: dict):
    with gr.Tab("MultiModal Models", id="multimodal_tab") as multimodal_tab:
        select_tab = params["default_second_tab"] if params["default_first_tab"] == "multimodal_tab" else None
        with gr.Tabs(selected=select_tab):
            multimodal_llm.create_ui(params)

    multimodal_tab.select(reload_default_model)
