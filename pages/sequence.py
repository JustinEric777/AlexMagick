import gradio as gr
from pages.sequences import llm, mt


def on_select(params, evt: gr.EventData):
    if isinstance(evt, str):
        selected = evt
    else:
        selected = evt.target._constructor_args[0]['id']
    if selected == "llm_tab":
        params["task_type"] = "sequence-llm"
        llm.init_model(params)
    elif selected == "mt_tab":
        params["task_type"] = "sequence-mt"
        mt.init_model(params)
    return


def create_ui(params: dict):
    with gr.Tab(label="Sequence Models", id="sequence_tab"):
        with gr.Tabs(selected=params["default_second_tab"]):
            with gr.Tab(label="LLM Model", id="llm_tab") as llm_tab:
                llm.create_ui(params)
            with gr.Tab(label="MT Model", id="mt_tab") as mt_tab:
                mt.create_ui(params)

            hidden_params = gr.State(params)
            on_select(params, params["default_second_tab"])
            llm_tab.select(on_select, hidden_params)
            mt_tab.select(on_select, hidden_params)


