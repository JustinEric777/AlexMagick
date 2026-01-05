import gradio as gr
from pages.common import create_standard_page
from servers import mt_server

mt = mt_server.MtServer()


def create_ui(args: dict):
    components = {}
    
    def render_main():
        with gr.Row():
            components['text_input'] = gr.Textbox(label="MT Input", lines=10, placeholder="Original Text...",)
            components['text_output'] = gr.Textbox(label="MT Output", lines=10, placeholder="Translated Text...",)
        with gr.Row():
            components['translate_bt'] = gr.Button("Translate", variant="primary")
            components['clear'] = gr.Button("Clear")
            components['metric'] = gr.Textbox(label="Metric Info", placeholder="metric info...", visible=False)
        with gr.Row():
            components['results'] = gr.Dataframe(
                label="Translate Results",
                headers=["Original Text", "Translate Text", "Metric"],
                datatype="markdown",
                column_widths=[40, 40, 20],
                wrap=True
            )

        def update_results(original_text, translated_text, metric_value):
            items = components['results'].value["data"]
            new_row = [original_text, translated_text, metric_value.strip()]
            items.append(new_row)
            return items

        components['translate_bt'].click(
            mt.generate, 
            inputs=[components['text_input']], 
            outputs=[components['text_output'], components['metric']], 
            queue=False
        ).then(
            update_results,  
            inputs=[components['text_input'], components['text_output'], components['metric']], 
            outputs=[components['results']], 
            queue=False
        )

        components['clear'].click(lambda: "", None, [components['text_input'], components['text_output']], queue=False)

    create_standard_page(mt, args, render_main, tab_label="MT Model", tab_id="mt_tab")
