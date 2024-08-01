import gradio as gr
from pages.common import reload_model_ui
from modules import llm


def create_ui(args: dict):
    llm.init_model(args)

    with gr.Tab(label="LLM Model", id="llm_tab") as llm_tab:
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(height=500)
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
                            ]
                        ],
                        inputs=[msg],
                    )
                with gr.Row():
                    clear = gr.Button("New Topic")
                    re_generate = gr.Button("Regenerate")
                    sent_bt = gr.Button("Send", variant="primary")
            with gr.Column(scale=1):
                infer_arch, model_name, model_version = reload_model_ui(llm, args)
                with gr.Row():
                    with gr.Accordion("generate params", open=True):
                        max_tokens = gr.Slider(minimum=512, maximum=4096, label="max_tokens", value=2048)
                        slider_temp = gr.Slider(minimum=0, maximum=1, label="temperature", value=0.6)
                        slider_top_p = gr.Slider(minimum=0.5, maximum=1, label="top_p", value=0.95)
                        slider_context_times = gr.Slider(minimum=0, maximum=5, label="context times", value=2, step=2.0)

        def user(user_message, history):
            return "", history + [[user_message, None]]

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=True).then(
            llm.generate,
            [chatbot, max_tokens, slider_temp, slider_top_p, slider_context_times],
            chatbot
        )
        sent_bt.click(user, [msg, chatbot], [msg, chatbot], queue=True).then(
            llm.generate,
            [chatbot, max_tokens, slider_temp, slider_top_p, slider_context_times],
            chatbot
        )
        re_generate.click(
            llm.generate,
            [chatbot, max_tokens, slider_temp, slider_top_p, slider_context_times],
            chatbot
        )
        clear.click(lambda: [], None, chatbot, queue=True)

    llm_tab.select(llm.reload_model, [infer_arch, model_name, model_version], [infer_arch, model_name, model_version])
