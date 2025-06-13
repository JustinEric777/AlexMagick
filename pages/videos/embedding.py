import gradio as gr
from pages.common import reload_model_ui
from modules import video2embedding


def create_ui(args: dict):
    video2embedding.init_model(args)

    with gr.Tab(label="Video2Embedding Model", id="video2embedding_tab") as mt_tab:
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Row():
                    text_inputs = gr.Textbox(label="Texts Input", lines=10, placeholder="Texts Input - Split based on newlines ...",)
                    search_input = gr.Textbox(label="Search Input", lines=10, placeholder="Search Text ...",)
                    search_result = gr.Textbox(label="Search Input", visible=False, lines=10, placeholder="Search Result ...",)
                with gr.Row():
                    search_bt = gr.Button("Search", variant="primary")
                    clear = gr.Button("Clear")
                    metric = gr.Textbox(label="Metric Info", placeholder="metric info...", visible=False)
                with gr.Row():
                    with gr.Column(scale=4):
                        gr.Examples(
                            label="Input Examples",
                            examples=[
                                [
                                    """what is the capital of China?
how to implement quick sort in python?
北京
快排算法介绍""",
                                    "快速排序"
                                ],
                            ],
                            elem_id="text2embedding_examples",
                            inputs=[text_inputs, search_input],
                        )
                with gr.Row():
                    results = gr.Dataframe(
                        label="Search Results",
                        headers=["Search Input", "Texts", "Search Result", "Metric"],
                        datatype="markdown",
                        column_widths=[20, 40, 20, 20],
                        wrap=True
                    )
            with gr.Column(scale=1):
                infer_arch, device, model_name, model_version = reload_model_ui(video2embedding, args)

        def update_results(input_texts, search_text, search_results, metric_value):
            items = results.value["data"]
            new_row = [input_texts, search_text, search_results, metric_value.strip()]
            items.append(new_row)
            return items

        search_bt.click(video2embedding.generate, inputs=[text_inputs, search_input, model_version], outputs=[search_result, metric], queue=False).then(
            update_results,  inputs=[search_input, text_inputs, search_result, metric], outputs=[results], queue=False
        )

        clear.click(lambda: "", None, [text_inputs, search_input, search_result], queue=False)

    mt_tab.select(video2embedding.reload_model, [infer_arch, device, model_name, model_version], [infer_arch, device, model_name, model_version])

