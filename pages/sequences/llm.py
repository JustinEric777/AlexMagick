import gradio as gr
from pages.common import reload_model_ui
from modules import llm


def create_ui(args: dict):
    llm.init_model(args)

    with gr.Tab(label="LLM Model", id="llm_tab") as llm_tab:
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot()
                msg = gr.Textbox(label="Chatbot Input", lines=5, placeholder="Shift + Enter Send Message...", )
                with gr.Row():
                    clear = gr.Button("New Topic")
                    re_generate = gr.Button("Regenerate")
                    sent_bt = gr.Button("Send", variant="primary")
            with gr.Column(scale=1):
                infer_arch, model_name, model_version = reload_model_ui(llm, args)

                with gr.Row():
                    with gr.Accordion("generate params", open=True):
                        slider_temp = gr.Slider(minimum=0, maximum=1, label="temperature", value=0.6)
                        slider_top_p = gr.Slider(minimum=0.5, maximum=1, label="top_p", value=0.95)
                        slider_context_times = gr.Slider(minimum=0, maximum=5, label="context times", value=0, step=2.0)

        def user(user_message, history):
            return "", history + [[user_message, None]]

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            llm.generate,
            [chatbot, slider_temp, slider_top_p, slider_context_times, infer_arch, model_name],
            chatbot
        )
        sent_bt.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            llm.generate,
            [chatbot, slider_temp, slider_top_p, slider_context_times, infer_arch, model_name],
            chatbot
        )
        re_generate.click(
            llm.generate,
            [chatbot, slider_temp, slider_top_p, slider_context_times,  infer_arch, model_name],
            chatbot
        )
        clear.click(lambda: [], None, chatbot, queue=False)

    llm_tab.select(llm.reload_model, [infer_arch, model_name, model_version], [infer_arch, model_name, model_version])
