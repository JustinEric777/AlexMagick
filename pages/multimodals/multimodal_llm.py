import gradio as gr
from pages.common import create_standard_page
from servers import multimodal_llm_server

multimodal_llm = multimodal_llm_server.MultimodalLLMServer()

def create_ui(args: dict):
    components = {}

    def render_main():
        components['chatbot'] = gr.Chatbot(height=500)
        components['msg'] = gr.MultimodalTextbox(
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
                inputs=[components['msg']],
            )

    def render_params():
        with gr.Accordion("generate params", open=True):
            components['return_audio'] = gr.Radio(choices=[("Yes", True), ("No", False)], value=True, interactive=True, label="return_audio")
            components['max_tokens'] = gr.Slider(minimum=512, maximum=4096, label="max_tokens", value=2048)
            components['slider_temp'] = gr.Slider(minimum=0, maximum=1, label="temperature", value=0.6)
            components['slider_top_p'] = gr.Slider(minimum=0.5, maximum=1, label="top_p", value=0.95)
            components['slider_context_times'] = gr.Slider(minimum=0, maximum=5, label="context times", value=0, step=2.0)

    create_standard_page(multimodal_llm, args, render_main, render_params_ui=render_params, tab_label="MultiModal LLM Model", tab_id="mllm_tab")

    def user(message, history):
        if history is None: history = []
        
        # Add files as separate messages
        for x in message["files"]:
            history.append({"role": "user", "content": x})
        
        if message["text"]:
            history.append({"role": "user", "content": message["text"]})

        return gr.MultimodalTextbox(value=None, interactive=False), history

    def generate_wrapper(history, max_tokens, temperature, top_p, slider_context_times, return_audio):
        server_messages = []
        for msg in history:
            content = msg["content"]
            role = msg["role"]
            
            if isinstance(content, str) and (content.endswith('.jpg') or content.endswith('.jpeg') or content.endswith('.png') or content.endswith('.wav') or content.endswith('.mp4')):
                    server_messages.append({"role": role, "content": {"path": content}})
            else:
                    server_messages.append({"role": role, "content": content})

        for server_history in multimodal_llm.generate(server_messages, max_tokens, temperature, top_p, slider_context_times, return_audio):
            ui_history = []
            for msg in server_history:
                content = msg["content"]
                role = msg["role"]
                
                if isinstance(content, dict) and "path" in content:
                    ui_content = content["path"]
                else:
                    ui_content = content
                
                ui_history.append({"role": role, "content": ui_content})
            
            yield ui_history

    components['msg'].submit(user, [components['msg'], components['chatbot']], [components['msg'], components['chatbot']], queue=True).then(
        generate_wrapper,
        [components['chatbot'], components['max_tokens'], components['slider_temp'], components['slider_top_p'], components['slider_context_times'], components['return_audio']],
        components['chatbot'],
        queue=True
    ).then(lambda: gr.MultimodalTextbox(interactive=True), None, [components['msg']])
