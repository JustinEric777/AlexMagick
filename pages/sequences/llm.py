import gradio as gr
from pages.common import reload_model_ui
from modules import llm


def create_ui(args: dict):
    llm.init_model(args)

    with gr.Tab(label="LLM Model", id="llm_tab") as llm_tab:
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(height=500, type='messages', show_copy_button=True)
                msg = gr.Textbox(label="Chatbot Input", lines=5, placeholder="Shift + Enter Send Message...", )
                with gr.Row():
                    gr.Examples(
                        label="Input Examples",
                        examples=[
                            [
                                "左手一只鸭，右手一只鸡。交换两次后左右手里各是什么？",
                            ],
                            [
                                "鸡兔同笼，共35只头，94只脚，问鸡兔各多少？",
                            ],
                            [
                                "Strawberry里有几个r？",
                            ],
                            [
                                "一只死猫与核同位素、一瓶毒药和辐射探测器一起放入盒子中。如果辐射探测器检测到辐射，它将释放毒药。一天后，盒子打开。猫还活着吗？",
                            ],
                            [
                                "假设在一个电车轨道上被绑了5个人，而它的备用轨道上被绑了1个人，又有一辆失控的电车飞速驶来，而你身边正好有一个摇杆，你可以推动摇杆来让电车驶入备用轨道。你该怎么做起伤害最小？",
                            ],
                            [
                                "为什么会有这样的语言现象:生鱼片是死鱼片，等红灯是在等绿灯，咖啡因来自咖啡果，救火是在灭火，晒太阳是在晒人，肉夹馍是馍夹肉"
                            ]
                        ],
                        inputs=[msg],
                    )
                with gr.Row():
                    clear = gr.Button("New Topic")
                    re_generate = gr.Button("Regenerate")
                    sent_bt = gr.Button("Send", variant="primary")
            with gr.Column(scale=1):
                infer_arch, device, model_name, model_version = reload_model_ui(llm, args)
                with gr.Row():
                    with gr.Accordion("generate params", open=True):
                        max_tokens = gr.Slider(minimum=512, maximum=4096, label="max_tokens", value=2048)
                        slider_temp = gr.Slider(minimum=0, maximum=1, label="temperature", value=0.6)
                        slider_top_p = gr.Slider(minimum=0.5, maximum=1, label="top_p", value=0.95)
                        slider_context_times = gr.Slider(minimum=0, maximum=5, label="context times", value=0, step=2.0)

        def user(user_message, history):
            return "", history + [{"role": "user", "content": user_message}]

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=True).then(
            llm.generate,
            [chatbot, max_tokens, slider_temp, slider_top_p, slider_context_times],
            chatbot,
            queue=True
        )
        sent_bt.click(user, [msg, chatbot], [msg, chatbot], queue=True).then(
            llm.generate,
            [chatbot, max_tokens, slider_temp, slider_top_p, slider_context_times],
            chatbot,
            queue=True
        )
        re_generate.click(
            llm.generate,
            [chatbot, max_tokens, slider_temp, slider_top_p, slider_context_times],
            chatbot,
            queue=True
        )
        clear.click(lambda: [], None, chatbot, queue=True)

    llm_tab.select(llm.reload_model, [infer_arch, device, model_name, model_version], [infer_arch, device, model_name, model_version])
