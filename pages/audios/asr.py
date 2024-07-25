import gradio as gr
from pages.common import reload_model_ui, HOST_PREFIX
from modules import asr


def create_ui(args: dict):
    asr.init_model(args)

    with gr.Tab("ASR Model", id="asr_tab") as asr_tab:
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Row():
                    input_audio = gr.Audio(
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
                    text_output = gr.Textbox(label="VSR Output", lines=10, placeholder="Transcribe Text...", )
                with gr.Row():
                    generate_bt = gr.Button("Generate", variant="primary")
                    clear = gr.Button("Clear")
                    metric = gr.Textbox(visible=False)
                with gr.Row():
                    results = gr.Dataframe(
                        label="Transcribe Results",
                        headers=["Original Audio", "Transcribe Text", "Metric"],
                        datatype="markdown",
                        column_widths=[40, 40, 20],
                        wrap=True
                    )
            with gr.Column(scale=1):
                infer_arch, model_name, model_version = reload_model_ui(asr, args)

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

        generate_bt.click(asr.generate, inputs=[input_audio, model_version], outputs=[text_output, metric], queue=False).then(
            update_results, inputs=[input_audio, text_output, metric], outputs=[results], queue=False
        )

        clear.click(lambda: None, None, [input_audio, text_output, results], queue=False)

    asr_tab.select(asr.reload_model, [infer_arch, model_name, model_version], [infer_arch, model_name, model_version])
