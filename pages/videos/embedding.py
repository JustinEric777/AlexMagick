import gradio as gr
from pages.common import create_standard_page
from servers.video2embedding_server import Video2EmbeddingServer

video2embedding = Video2EmbeddingServer()

def create_ui(args: dict):
    components = {}

    def render_main():
        with gr.Row():
            components['text_inputs'] = gr.Textbox(label="Texts Input", lines=10, placeholder="Texts Input - Split based on newlines ...",)
            components['search_input'] = gr.Textbox(label="Search Input", lines=10, placeholder="Search Text ...",)
            components['search_result'] = gr.Textbox(label="Search Input", visible=False, lines=10, placeholder="Search Result ...",)
        with gr.Row():
            components['search_bt'] = gr.Button("Search", variant="primary")
            components['clear'] = gr.Button("Clear")
            components['metric'] = gr.Textbox(label="Metric Info", placeholder="metric info...", visible=False)
        with gr.Row():
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
                inputs=[components['text_inputs'], components['search_input']],
            )
        with gr.Row():
            components['results'] = gr.Dataframe(
                label="Search Results",
                headers=["Search Input", "Texts", "Search Result", "Metric"],
                datatype="markdown",
                column_widths=[20, 40, 20, 20],
                wrap=True
            )

    create_standard_page(video2embedding, args, render_main, tab_label="Video2Embedding Model", tab_id="video2embedding_tab")

    def update_results(input_texts, search_text, search_results, metric_value):
        items = components['results'].value["data"]
        new_row = [input_texts, search_text, search_results, metric_value.strip()]
        items.append(new_row)
        return items

    components['search_bt'].click(
        video2embedding.generate, 
        inputs=[components['text_inputs'], components['search_input']], 
        outputs=[components['search_result'], components['metric']], 
        queue=False
    ).then(
        update_results,  
        inputs=[components['search_input'], components['text_inputs'], components['search_result'], components['metric']], 
        outputs=[components['results']], 
        queue=False
    )

    components['clear'].click(lambda: "", None, [components['text_inputs'], components['search_input'], components['search_result']], queue=False)
