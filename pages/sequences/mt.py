import gradio as gr


def create_ui(args: dict):
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Row():
                text_input = gr.Textbox(label="MT Input", lines=10, placeholder="Original Text...",)
                text_output = gr.Textbox(label="MT Output", lines=10, placeholder="Translated Text...",)
            with gr.Row():
                translate_bt = gr.Button("Translate", variant="primary")
                clear = gr.Button("Clear")
            with gr.Row():
                results = gr.Dataframe(
                    label="Translate Results",
                    headers=["Original Text", "Translate Text", "Metric"],
                )
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
                        slider_model_reload = gr.Button("Load Model...", variant="primary")
            with gr.Row():
                with gr.Accordion("Generate parameters", open=True):
                    slider_temp = gr.Slider(minimum=0, maximum=1, label="temperature", value=0.6)
                    slider_top_p = gr.Slider(minimum=0.5, maximum=1, label="top_p", value=0.95)
                    slider_context_times = gr.Slider(minimum=0, maximum=5, label="上文轮次", value=0, step=2.0)

