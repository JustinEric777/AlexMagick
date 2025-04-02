import gradio as gr
from pages.common import reload_model_ui, HOST_PREFIX
from modules import tts


def create_ui(args: dict):
    tts.init_model(args)

    with gr.Tab("TTS Model", id="tts_tab") as asr_tab:
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Row():
                    text_input = gr.Textbox(label="TTS Input", lines=10, placeholder="Input Text...", )
                    audio_output = gr.Audio(
                        label="Audio",
                        type="filepath",
                        sources=["upload", "microphone"],
                        waveform_options=gr.WaveformOptions(
                            waveform_color="#01C6FF",
                            waveform_progress_color="#0066B4",
                            skip_length=2,
                            show_controls=False,
                        ),
                    )
                with gr.Row():
                    generate_bt = gr.Button("Generate", variant="primary")
                    clear = gr.Button("Clear")
                    metric = gr.Textbox(visible=False)
                with gr.Row():
                    results = gr.Dataframe(
                        label="TTS Results",
                        headers=["Original Text", "Generate Audio", "Metric"],
                        datatype="markdown",
                        column_widths=[40, 40, 20],
                        wrap=True
                    )
            with gr.Column(scale=1):
                infer_arch, device, model_name, model_version = reload_model_ui(tts, args)

        def update_results(original_audio, translated_text, metric_value):
            items = results.value["data"]
            audio = f""" <audio controls>
                              <source src="{HOST_PREFIX}{original_audio}" type="audio/wav">
                         </audio>"""
            new_row = [audio, translated_text, metric_value.strip()]
            if len(items[0][0]) == 0 and len(items[0][1]) == 0:
                items[0] = new_row
            else:
                items.append(new_row)
            return items

        generate_bt.click(tts.generate, inputs=[text_input, model_version], outputs=[audio_output, metric], queue=False).then(
            update_results, inputs=[text_input, audio_output, metric], outputs=[results], queue=False
        )

        clear.click(lambda: None, None, [text_input, audio_output, results], queue=False)

    asr_tab.select(tts.reload_model, [infer_arch, device, model_name, model_version], [infer_arch, device, model_name, model_version])
