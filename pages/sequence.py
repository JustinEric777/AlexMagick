import gradio as gr
from pages.sequences import llm, mt, embedding


def reload_default_model():
    llm.llm.reload_model(default=True)


def create_ui(params: dict):
    with gr.Tab(label="Sequence Models", id="sequence_tab") as sequence_tab:
        with gr.Tabs(selected=params["default_second_tab"]):
            llm.create_ui(params)
            mt.create_ui(params)
            embedding.create_ui(params)

    sequence_tab.select(reload_default_model)

