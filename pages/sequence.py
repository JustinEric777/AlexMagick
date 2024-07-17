import gradio as gr
from pages.sequences import llm, mt


def create_ui(params: dict):
    with gr.Tab(label="Sequence Models", id="sequence_tab") as sequence_tab:
        with gr.Tabs(selected=params["default_second_tab"]):
            llm.create_ui(params)
            mt.create_ui(params)

    sequence_tab.select(llm.reload_model)

