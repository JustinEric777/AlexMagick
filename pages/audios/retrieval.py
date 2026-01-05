import gradio as gr
from pages.common import create_standard_page, create_audio_player, HOST_PREFIX
from servers import audio2embedding_server

audio2embedding = audio2embedding_server.AudioEmbeddingServer()


def create_ui(args: dict):
    components = {}

    def render_main():
        with gr.Row():
            components['text_inputs'] = gr.Textbox(label="Texts Input", lines=10, placeholder="Texts Input - Split based on newlines ...",)
            components['input_audios'] = gr.Audio(
                label="Search Audio",
                type="filepath",
                sources=["upload"],
                waveform_options=gr.WaveformOptions(
                    waveform_color="#01C6FF",
                    waveform_progress_color="#0066B4",
                    skip_length=2,
                ),
            )
            components['search_result'] = gr.Textbox(label="Search Results", visible=False, lines=10, placeholder="Search Result ...",)
        with gr.Row():
            components['search_bt'] = gr.Button("Search", variant="primary")
            components['clear'] = gr.Button("Clear")
            components['metric'] = gr.Textbox(label="Metric Info", placeholder="metric info...", visible=False)
        with gr.Row():
            gr.Examples(
                label="Input Examples",
                examples=[
                    [
                        """Sound of a dog.
Sound of vaccum cleaner""",
                        "./pages/examples/audios/audio.wav"
                    ],
                    [
                        """Sound of a dog.
Sound of vaccum cleaner""",
                        "./pages/examples/audios/1-02董卓瑶.mp3"
                    ],
                ],
                elem_id="audio2embedding_examples",
                inputs=[components['text_inputs'], components['input_audios']],
            )
        with gr.Row():
            components['results'] = gr.Dataframe(
                label="Search Results",
                headers=["Search Input", "Texts", "Search Result", "Metric"],
                datatype="markdown",
                column_widths=[20, 40, 20, 20],
                wrap=True
            )

    create_standard_page(audio2embedding, args, render_main, tab_label="Audio Retrieval Model", tab_id="audio_retrieval_tab")

    def update_results(input_texts, search_audio, search_results, metric_value):
        items = components['results'].value["data"]
        audio = create_audio_player(search_audio)
        new_row = [input_texts, audio, search_results, metric_value.strip()]
        items.append(new_row)
        return items

    components['search_bt'].click(
        audio2embedding.generate, 
        inputs=[components['text_inputs'], components['input_audios']], 
        outputs=[components['search_result'], components['metric']], 
        queue=False
    ).then(
        update_results,  
        inputs=[components['text_inputs'], components['input_audios'], components['search_result'], components['metric']], 
        outputs=[components['results']], 
        queue=False
    )

    components['clear'].click(
        lambda: "", 
        None, 
        [components['text_inputs'], components['input_audios'], components['search_result']], 
        queue=False
    )
