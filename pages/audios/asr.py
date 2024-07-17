import gradio as gr
from typing import Union
from modules import asr
from modules.servers.asr_server import get_model_list, TASK_TYPE


def init_model(params: dict):
    asr.load_model(params["task_type"], params["model_name"])


def reload_model(model_name: Union[None | str]):
    return asr.load_model(TASK_TYPE, model_name)


def create_ui(args: dict):
    init_model(args)

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
                with gr.Row():
                    with gr.Accordion("model setting", open=True):
                        model_name = gr.Dropdown(
                            label="models",
                            info="Please select the model to be infer...",
                            choices=get_model_list(),
                            value=args["model_name"] if args["model_name"] in get_model_list() else get_model_list()[0],
                            interactive=True
                        )
                with gr.Row():
                    model_reload_bt = gr.Button("Load Model...", variant="primary")

        def update_results(original_audio, translated_text, metric_value):
            items = results.value["data"]
            audio = f""" <audio controls>
                              <source src="{original_audio}" type="audio/wav">
                         </audio>"""
            print(audio)
            new_row = [audio, translated_text, metric_value.strip()]
            if len(items[0][0]) == 0 and len(items[0][1]) == 0:
                items[0] = new_row
            else:
                items.append(new_row)
            return items

        generate_bt.click(asr.generate, inputs=[input_audio, model_name], outputs=[text_output, metric], queue=False).then(
            update_results, inputs=[input_audio, text_output, metric], outputs=[results], queue=False
        )
        model_reload_bt.click(reload_model, [model_name], [model_name], show_progress="full")
        clear.click(lambda: None, None, [input_audio, text_output, results], queue=False)

    asr_tab.select(reload_model, [model_name], [model_name])

