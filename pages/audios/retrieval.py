import gradio as gr
from pages.common import reload_model_ui, HOST_PREFIX
from modules import audio2embedding


def create_ui(args: dict):
    audio2embedding.init_model(args)

    with gr.Tab(label="Retrieval Model", id="audio2embedding_tab") as mt_tab:
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Row():
                    text_inputs = gr.Textbox(label="Texts Input", lines=10, placeholder="Texts Input - Split based on newlines ...",)
                    input_audios = gr.Audio(
                        label="Search Audio",
                        type="filepath",
                        sources=["upload"],
                        waveform_options=gr.WaveformOptions(
                            waveform_color="#01C6FF",
                            waveform_progress_color="#0066B4",
                            skip_length=2,
                            show_controls=False,
                        ),
                    )
                    search_result = gr.Textbox(label="Search Results", visible=False, lines=10, placeholder="Search Result ...",)
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
                            inputs=[text_inputs, input_audios],
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
                infer_arch, device, model_name, model_version = reload_model_ui(audio2embedding, args)

        def update_results(input_texts, search_audio, search_results, metric_value):
            items = results.value["data"]
            audio = f""" <audio controls>
                              <source src="{HOST_PREFIX}{search_audio}" type="audio/wav">
                         </audio>"""
            new_row = [input_texts, audio, search_results, metric_value.strip()]
            items.append(new_row)
            return items

        search_bt.click(audio2embedding.generate, inputs=[text_inputs, input_audios, model_version], outputs=[search_result, metric], queue=False).then(
            update_results,  inputs=[text_inputs, input_audios, search_result, metric], outputs=[results], queue=False
        )

        clear.click(lambda: "", None, [text_inputs, input_audios, search_result], queue=False)

    mt_tab.select(audio2embedding.reload_model, [infer_arch, device, model_name, model_version], [infer_arch, device, model_name, model_version])

