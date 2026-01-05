import gradio as gr
from pages.common import create_standard_page, create_audio_player, HOST_PREFIX
from servers import asr_server

asr = asr_server.ASRServer()


def create_ui(args: dict):
    def render_main():
        with gr.Row():
            input_audio = gr.Audio(
                label="Audio",
                type="filepath",
                sources=["upload", "microphone"],
                waveform_options=gr.WaveformOptions(
                    waveform_color="#01C6FF",
                    waveform_progress_color="#0066B4",
                    skip_length=2,
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

        def update_results(original_audio, translated_text, metric_value):
            items = results.value["data"]
            audio = create_audio_player(original_audio)
            new_row = [audio, translated_text, metric_value.strip()]
            items.append(new_row)
            return items

        generate_bt.click(asr.generate, inputs=[input_audio], outputs=[text_output, metric], queue=False).then(
            update_results, inputs=[input_audio, text_output, metric], outputs=[results], queue=False
        )

        clear.click(lambda: None, None, [input_audio, text_output, results], queue=False)

    create_standard_page(asr, args, render_main, tab_label="ASR Model", tab_id="asr_tab")
