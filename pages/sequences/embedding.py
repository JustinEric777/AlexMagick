import gradio as gr
from pages.common import reload_model_ui
from modules import mt


def create_ui(args: dict):
    mt.init_model(args)

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
                infer_arch, model_name, model_version = reload_model_ui(mt, args)

        def update_results(original_text, translated_text, metric_value):
            items = results.value["data"]
            new_row = [original_text, translated_text, metric_value.strip()]
            if len(items[0][0]) == 0 and len(items[0][1]) == 0:
                items[0] = new_row
            else:
                items.append(new_row)
            return items

        translate_bt.click(mt.generate, inputs=[text_input, model_version], outputs=[text_output, metric], queue=False).then(
            update_results,  inputs=[text_input, text_output, metric], outputs=[results], queue=False
        )

        clear.click(lambda: "", None, [text_input, text_output], queue=False)

    mt_tab.select(mt.reload_model, [infer_arch, model_name, model_version], [infer_arch, model_name, model_version])

