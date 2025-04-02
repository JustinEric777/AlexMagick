import gradio as gr
from gradio_multimodalchatbot import MultimodalChatbot
from gradio.data_classes import FileData
from pages.common import reload_model_ui
from modules import multimodal_llm

user_msg1 = {"text": "Hello, what is in this image?",
             "files": [{"file": FileData(path="https://gradio-builds.s3.amazonaws.com/diffusion_image/cute_dog.jpg")}]
             }
bot_msg1 = {"text": "It is a very cute dog",
            "files": []}

user_msg2 = {"text": "Describe this audio clip please.",
             "files": [{"file": FileData(path="cantina.wav")}]}
bot_msg2 = {"text": "It is the cantina song from Star Wars",
            "files": []}

user_msg3 = {"text": "Give me a video clip please.",
             "files": []}
bot_msg3 = {"text": "Here is a video clip of the world",
            "files": [{"file": FileData(path="world.mp4")},
                      {"file": FileData(path="cantina.wav")}]}

conversation = [[user_msg1, bot_msg1], [user_msg2, bot_msg2], [user_msg3, bot_msg3]]


def create_ui(args: dict):
    multimodal_llm.init_model(args)

    with gr.Tab(label="MultiModal LLM Model", id="mllm_tab") as mllm_tab:
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = MultimodalChatbot(value=conversation, height=500)
                msg = gr.MultimodalTextbox(
                    label="Chatbot Input",
                    lines=1,
                    interactive=True,
                    file_count="multiple",
                    placeholder="Enter message or upload file，Shift + Enter Send Message...",
                    show_label=False,
                    sources=["microphone", "upload"],
                )
                with gr.Row():
                    gr.Examples(
                        label="Input Examples",
                        examples=[
                            [
                                "左手一只鸭，右手一只鸡。交换两次后左右手里各是什么？",
                            ],
                        ],
                        inputs=[msg],
                    )
                with gr.Row():
                    clear = gr.Button("New Topic")
                    re_generate = gr.Button("Regenerate")
                    sent_bt = gr.Button("Send", variant="primary")
            with gr.Column(scale=1):
                infer_arch, device, model_name, model_version = reload_model_ui(multimodal_llm, args)
                with gr.Row():
                    with gr.Accordion("generate params", open=True):
                        max_tokens = gr.Slider(minimum=512, maximum=4096, label="max_tokens", value=2048)
                        slider_temp = gr.Slider(minimum=0, maximum=1, label="temperature", value=0.6)
                        slider_top_p = gr.Slider(minimum=0.5, maximum=1, label="top_p", value=0.95)
                        slider_context_times = gr.Slider(minimum=0, maximum=5, label="context times", value=0, step=2.0)

        def user(user_message, history):
            return "", history + [[user_message, None]]

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=True).then(
            multimodal_llm.generate,
            [chatbot, max_tokens, slider_temp, slider_top_p, slider_context_times],
            chatbot,
            queue=True
        )
        sent_bt.click(user, [msg, chatbot], [msg, chatbot], queue=True).then(
            multimodal_llm.generate,
            [chatbot, max_tokens, slider_temp, slider_top_p, slider_context_times],
            chatbot,
            queue=True
        )
        re_generate.click(
            multimodal_llm.generate,
            [chatbot, max_tokens, slider_temp, slider_top_p, slider_context_times],
            chatbot,
            queue=True
        )
        clear.click(lambda: [], None, chatbot, queue=True)

    mllm_tab.select(multimodal_llm.reload_model, [infer_arch, device, model_name, model_version], [infer_arch, device, model_name, model_version])
