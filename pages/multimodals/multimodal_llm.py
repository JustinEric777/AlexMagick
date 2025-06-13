import gradio as gr
from pages.common import reload_model_ui
from modules import multimodal_llm


def create_ui(args: dict):
    multimodal_llm.init_model(args)

    with gr.Tab(label="MultiModal LLM Model", id="mllm_tab") as mllm_tab:
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(height=500, type='messages', show_copy_button=True)
                msg = gr.MultimodalTextbox(
                    label="Chatbot Input",
                    lines=1,
                    interactive=True,
                    file_count="multiple",
                    placeholder="Enter message or upload file...",
                    show_label=False,
                    sources=["microphone", "upload"],
                    stop_btn=True,
                )
                with gr.Row():
                    gr.Examples(
                        label="Input Examples",
                        examples=[
                            [
                                {
                                    'text': '描述一下这张图片内容',
                                    'files': ['./pages/examples/multimodals/image_1.jpeg']
                                }
                            ],
                            [
                                {
                                    'text': '描述一下这段语音讲了什么？',
                                    'files': ['./pages/examples/multimodals/test_zh.wav']
                                }
                            ],
                            [
                                {
                                    'text': '描述一下这段8s视频的内容？',
                                    'files': ['./pages/examples/multimodals/test_2.mp4']
                                }
                            ],
                            [
                                {
                                    'text': '描述一下这段30s视频的内容？',
                                    'files': ['./pages/examples/multimodals/test_ko.mp4']
                                }
                            ],
                        ],
                        inputs=[msg],
                    )
            with gr.Column(scale=1):
                infer_arch, device, model_name, model_version = reload_model_ui(multimodal_llm, args)
                with gr.Row():
                    with gr.Accordion("generate params", open=True):
                        return_audio = gr.Radio(choices=[("Yes", True), ("No", False)], value=True, interactive=True, label="return_audio")
                        max_tokens = gr.Slider(minimum=512, maximum=4096, label="max_tokens", value=2048)
                        slider_temp = gr.Slider(minimum=0, maximum=1, label="temperature", value=0.6)
                        slider_top_p = gr.Slider(minimum=0.5, maximum=1, label="top_p", value=0.95)
                        slider_context_times = gr.Slider(minimum=0, maximum=5, label="context times", value=0, step=2.0)

        def user(message, history):
            for x in message["files"]:
                history.append({"role": "user", "content": {"path": x}})
            if message["text"] is not None:
                history.append({"role": "user", "content": message["text"]})

            return gr.MultimodalTextbox(value=None, interactive=False), history

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=True).then(
            multimodal_llm.generate,
            [chatbot, max_tokens, slider_temp, slider_top_p, slider_context_times, return_audio],
            chatbot,
            queue=True
        ).then(lambda: gr.MultimodalTextbox(interactive=True), None, [msg])

    mllm_tab.select(multimodal_llm.reload_model, [infer_arch, device, model_name, model_version], [infer_arch, device, model_name, model_version])
