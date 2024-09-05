import gradio as gr
from pages.common import reload_model_ui
from modules import text2embedding


def create_ui(args: dict):
    text2embedding.init_model(args)

    with gr.Tab(label="Text2Embedding Model", id="text2embedding_tab") as mt_tab:
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Row():
                    text_input = gr.Textbox(label="Texts Input1", lines=10, placeholder="Original Texts...",)
                    text_output = gr.Textbox(label="Search Input", lines=10, placeholder="Search Text...",)
                with gr.Row():
                    search_bt = gr.Button("Search", variant="primary")
                    clear = gr.Button("Clear")
                    metric = gr.Textbox(label="Metric Info", placeholder="metric info...", visible=False)
                with gr.Row():
                    results = gr.Dataframe(
                        label="Search Results",
                        headers=["Search Input Text", "Search Result", "Metric"],
                        datatype="markdown",
                        column_widths=[40, 40, 20],
                        wrap=True
                    )
            with gr.Column(scale=1):
                infer_arch, model_name, model_version = reload_model_ui(text2embedding, args)

        def update_results(original_text, translated_text, metric_value):
            items = results.value["data"]
            new_row = [original_text, translated_text, metric_value.strip()]
            if len(items[0][0]) == 0 and len(items[0][1]) == 0:
                items[0] = new_row
            else:
                items.append(new_row)
            return items

        search_bt.click(text2embedding.generate, inputs=[text_input, model_version], outputs=[text_output, metric], queue=False).then(
            update_results,  inputs=[text_input, text_output, metric], outputs=[results], queue=False
        )

        clear.click(lambda: "", None, [text_input, text_output], queue=False)

    mt_tab.select(text2embedding.reload_model, [infer_arch, model_name, model_version], [infer_arch, model_name, model_version])

