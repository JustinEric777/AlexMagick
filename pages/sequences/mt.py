from typing import Union
import gradio as gr
from modules import mt
from modules.servers.mt_server import get_model_list, TASK_TYPE


def init_model(params: dict):
    mt.load_model(params["task_type"], params["model_name"])


def reload_model(model_name: Union[str | None]):
    return mt.load_model(TASK_TYPE, model_name)


def create_ui(args: dict):
    init_model(args)

    with gr.Tab(label="MT Model", id="mt_tab") as mt_tab:
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Row():
                    text_input = gr.Textbox(label="MT Input", lines=10, placeholder="Original Text...",)
                    text_output = gr.Textbox(label="MT Output", lines=10, placeholder="Translated Text...",)
                with gr.Row():
                    translate_bt = gr.Button("Translate", variant="primary")
                    clear = gr.Button("Clear")
                    metric = gr.Textbox(label="Metric Info", placeholder="metric info...", visible=False)
                with gr.Row():
                    results = gr.Dataframe(
                        label="Translate Results",
                        headers=["Original Text", "Translate Text", "Metric"],
                        datatype="markdown",
                        column_widths=[40, 40, 20],
                        wrap=True
                    )
            with gr.Column(scale=1):
                with gr.Row():
                    with gr.Accordion("model setting", open=True):
                        model_name = gr.Dropdown(
                            label="models",
                            info="Please select the model to be infer...",
                            choices=get_model_list(),
                            value=args["model_name"] if args["model_name"] in get_model_list() else get_model_list()[0],
                            interactive=True
                        )
                        with gr.Row():
                            model_reload_bt = gr.Button("Load Model...", variant="primary")

        def update_results(original_text, translated_text, metric_value):
            items = results.value["data"]
            new_row = [original_text, translated_text, metric_value.strip()]
            if len(items[0][0]) == 0 and len(items[0][1]) == 0:
                items[0] = new_row
            else:
                items.append(new_row)
            return items
        translate_bt.click(mt.generate, inputs=[text_input, model_name], outputs=[text_output, metric], queue=False).then(
            update_results,  inputs=[text_input, text_output, metric], outputs=[results], queue=False
        )
        model_reload_bt.click(reload_model, [model_name], [model_name], show_progress="full")
        clear.click(lambda: "", None, [text_input, text_output], queue=False)

    mt_tab.select(reload_model, [model_name], [model_name])
