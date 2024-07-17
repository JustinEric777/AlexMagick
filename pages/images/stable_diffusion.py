import gradio as gr
from typing import Union


def init_model(params: dict):
    print(params)


def reload_model(model_name: Union[str | None]):
    print(model_name)


def create_ui(args: dict):
    with gr.Tab(label="Image Generate Model", id="image_gen_tab") as image_gen_tab:
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    positive_prompt = gr.Textbox(label="Positive Prompt", lines=5, placeholder="Positive Prompt Text...",)
                with gr.Row():
                    negative_prompt = gr.Textbox(label="Negative Prompt", lines=5, placeholder="Negative Prompt Text...",)
                with gr.Row():
                    translate_bt = gr.Button("Generate", variant="primary")
                    clear = gr.Button("Clear")
                    metric = gr.Textbox(label="Metric Info", placeholder="metric info...", visible=False)
            with gr.Column(scale=2):
                with gr.Row():
                    image_output = gr.Image(label="Generated Output Image", height=380, interactive=False)
            with gr.Column(scale=1):
                with gr.Row():
                    with gr.Accordion("model setting", open=True):
                        model_name = gr.Dropdown(
                            label="models",
                            info="Please select the model to be infer...",
                            choices=[],
                            value=args["model_name"],
                            interactive=True
                        )
                        with gr.Row():
                            model_reload_bt = gr.Button("Load Model...", variant="primary")
