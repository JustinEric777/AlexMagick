import gradio as gr
from pages.common import reload_model_ui, HOST_PREFIX
from modules import tts


def create_ui(args: dict):
    tts.init_model(args)

    with gr.Tab("TTS Model", id="tts_tab") as tts_tab:
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
                    gr.Examples(
                        label="Input Examples",
                        examples=[
                            [
                                "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
                            ],
                            [
                                """第一章  陨落的天才()
        “斗之力，三段！”                       
        望着测验魔石碑上面闪亮得甚至有些刺眼的五个大字，少年面无表情，唇角有着一抹自嘲，紧握的手掌，因为大力，而导致略微尖锐的指甲深深的刺进了掌心之中，带来一阵阵钻心的疼痛…                          
        “萧炎，斗之力，三段！级别：低级！”测验魔石碑之旁，一位中年男子，看了一眼碑上所显示出来的信息，语气漠然的将之公布了出来…
        中年男子话刚刚脱口，便是不出意外的在人头汹涌的广场上带起了一阵嘲讽的骚动。
        “三段？嘿嘿，果然不出我所料，这个“天才”这一年又是在原地踏步！”
        “哎，这废物真是把家族的脸都给丢光了。”
        “要不是族长是他的父亲，这种废物，早就被驱赶出家族，任其自生自灭了，哪还有机会待在家族中白吃白喝。”
        """,
                            ],
                        ],
                        inputs=[text_input],
                    )
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
                with gr.Row():
                    with gr.Accordion("generate params", open=True):
                        max_tokens = gr.Slider(minimum=512, maximum=4096, label="max_tokens", value=2048)

        def update_results(original_text, generated_audio, metric_value):
            items = results.value["data"]
            audio = f""" <audio controls>
                              <source src="{HOST_PREFIX}{generated_audio}" type="audio/wav">
                         </audio>"""
            new_row = [original_text, audio, metric_value.strip()]
            items.append(new_row)

            return items

        generate_bt.click(tts.generate, inputs=[text_input, model_version], outputs=[audio_output, metric], queue=False).then(
            update_results, inputs=[text_input, audio_output, metric], outputs=[results], queue=False
        )

        clear.click(lambda: None, None, [text_input, audio_output, results], queue=False)

    tts_tab.select(tts.reload_model, [infer_arch, device, model_name, model_version], [infer_arch, device, model_name, model_version])
