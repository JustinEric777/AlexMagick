import gradio as gr
from pages.sequences import llm, mt


def create_ui(params: dict):
    with gr.Tab("Sequence Models", id="sequence_tab"):
        with gr.Tab("LLM Model", id="sequence_llm_tab"):
            llm.create_ui(params)
        with gr.Tab("MT Model", id="sequence_mt_tab"):
            mt.create_ui(params)

