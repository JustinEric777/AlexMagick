import gradio as gr
from pages.common import reload_model_ui
from servers import multimodal_llm_server

multimodal_llm = multimodal_llm_server.MultimodalLLMServer()

def create_ui(args: dict):
    with gr.Tab(label="MultiModal LLM Model", id="mllm_tab") as mllm_tab:
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(height=500)
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
            if history is None: history = []
            
            # Add files as separate messages
            for x in message["files"]:
                # For Gradio UI (type="messages"), we pass the path directly.
                # It will handle rendering.
                history.append({"role": "user", "content": x})
            
            if message["text"]:
                history.append({"role": "user", "content": message["text"]})

            return gr.MultimodalTextbox(value=None, interactive=False), history

        def generate_wrapper(history, max_tokens, temperature, top_p, slider_context_times, return_audio):
            # history is in UI format (List[Dict] with "content" as string or path string)
            
            # Convert UI History -> Server History
            # Server expects {"content": {"path": ...}} for files
            server_messages = []
            for msg in history:
                content = msg["content"]
                role = msg["role"]
                
                # Check if content is a file path
                if isinstance(content, str) and (content.endswith('.jpg') or content.endswith('.jpeg') or content.endswith('.png') or content.endswith('.wav') or content.endswith('.mp4')):
                     server_messages.append({"role": role, "content": {"path": content}})
                else:
                     server_messages.append({"role": role, "content": content})

            # Call generator
            for server_history in multimodal_llm.generate(server_messages, max_tokens, temperature, top_p, slider_context_times, return_audio):
                
                # Convert Server History -> UI History
                ui_history = []
                for msg in server_history:
                    content = msg["content"]
                    role = msg["role"]
                    
                    # If content is {"path": ...}, extract it for UI
                    if isinstance(content, dict) and "path" in content:
                        ui_content = content["path"]
                        # For audio returned by server, it might be in a separate message.
                        # We append it.
                    else:
                        ui_content = content
                    
                    ui_history.append({"role": role, "content": ui_content})
                
                yield ui_history

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=True).then(
            generate_wrapper,
            [chatbot, max_tokens, slider_temp, slider_top_p, slider_context_times, return_audio],
            chatbot,
            queue=True
        ).then(lambda: gr.MultimodalTextbox(interactive=True), None, [msg])

    mllm_tab.select(multimodal_llm.reload_model, [infer_arch, device, model_name, model_version], [infer_arch, device, model_name, model_version])
