import gradio as gr
import pandas as pd
from modules.mt_server import get_model_list, init_model, reload_model, translate


def create_ui(args: dict):
    init_model(args)

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
                    column_widths=[40, 40, 20],
                    line_breaks=True,
                    wrap=True
                )
        with gr.Column(scale=1):
            with gr.Row():
                with gr.Accordion("model setting", open=True):
                    model_name = gr.Dropdown(
                        label="models",
                        info="Please select the model to be infer...",
                        choices=get_model_list(),
                        value=get_model_list()[0],
                        interactive=True
                    )
                    with gr.Row():
                        model_reload_bt = gr.Button("Load Model...", variant="primary")
            with gr.Row():
                with gr.Accordion("Generate parameters", open=True):
                    slider_temp = gr.Slider(minimum=0, maximum=1, label="temperature", value=0.6)
                    slider_top_p = gr.Slider(minimum=0.5, maximum=1, label="top_p", value=0.95)
                    slider_context_times = gr.Slider(minimum=0, maximum=5, label="上文轮次", value=0, step=2.0)

    def update_results(original_text, translated_text):
        items = results.value["data"]
        new_row = [original_text, translated_text, gr.Json({"model_name": model_name.value})]
        if len(items[0][0]) == 0 and len(items[0][1]) == 0:
            items[0] = new_row
        else:
            items.append(new_row)
        return items
    translate_bt.click(translate, inputs=[text_input], outputs=[text_output], queue=False).then(
        update_results,  inputs=[text_input, text_output], outputs=[results], queue=False
    )
    model_reload_bt.click(reload_model, [model_name], [model_name], show_progress="full")
    clear.click(lambda: "", None, [text_input, text_output], queue=False)

