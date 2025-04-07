import gradio as gr
from pages.sequences import llm, mt, embedding


def reload_default_model():
    llm.llm.reload_model(default=True)


def create_ui(params: dict):
    with gr.Tab(label="Sequence Models", render=True, id="sequence_tab") as sequence_tab:
        select_tab = params["default_second_tab"] if params["default_first_tab"] == "sequence_tab" else None
        with gr.Tabs(selected=select_tab):
            llm.create_ui(params)
            mt.create_ui(params)
            embedding.create_ui(params)

    sequence_tab.select(reload_default_model)

