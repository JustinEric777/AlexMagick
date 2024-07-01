import gradio as gr
from modules.llm_server import get_inference_arch_list, get_model_arch_list, get_model_list, init_model,  reload_model, chat


def update_model_list(arch_model_name: str, infer_arch_name: str):
    model_list = get_model_list(arch_model_name, infer_arch_name)
    return gr.Dropdown(
        label="推理模型",
        info="请选择需要推理的模型",
        choices=model_list,
        value=model_list[0],
        interactive=True
    )


def create_ui(args: dict):
    init_model(args)

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot()
            msg = gr.Textbox(label="Chatbot Input", lines=5, placeholder="Shift + Enter Send Message...",)
            with gr.Row():
                clear = gr.Button("新话题")
                re_generate = gr.Button("重新回答")
                sent_bt = gr.Button("发送", variant="primary")
        with gr.Column(scale=1):
            with gr.Row():
                with gr.Accordion("模型推理设置", open=True):
                    arch_model = gr.Radio(
                        label="模型",
                        info="请选择模型",
                        choices=get_model_arch_list(),
                        value=args["arch_model"],
                        interactive=True
                    )
                    infer_arch = gr.Radio(
                        label="推理引擎",
                        info="请选择推理引擎",
                        choices=get_inference_arch_list(args["arch_model"]),
                        value=args["infer_arch"],
                        interactive=True
                    )
                    model_name = gr.Dropdown(
                        label="推理模型",
                        info="请选择需要推理的模型",
                        choices=get_model_list(args["arch_model"], args["infer_arch"]),
                        value=args["model_name"],
                        interactive=True
                    )
                    with gr.Row():
                        slider_model_reload = gr.Button("重新加载模型", variant="primary")
            with gr.Row():
                with gr.Accordion("生成参数", open=True):
                    slider_temp = gr.Slider(minimum=0, maximum=1, label="temperature", value=0.6)
                    slider_top_p = gr.Slider(minimum=0.5, maximum=1, label="top_p", value=0.95)
                    slider_context_times = gr.Slider(minimum=0, maximum=5, label="上文轮次", value=0, step=2.0)

    def user(user_message, history):
        return "", history + [[user_message, None]]

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        chat, [chatbot, slider_temp, slider_top_p, slider_context_times, infer_arch, model_name], chatbot
    )
    sent_bt.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        chat, [chatbot, slider_temp, slider_top_p, slider_context_times, infer_arch, model_name], chatbot
    )
    re_generate.click(chat, [chatbot, slider_temp, slider_top_p, slider_context_times, infer_arch, model_name], chatbot)
    clear.click(lambda: [], None, chatbot, queue=False)
    slider_model_reload.click(reload_model, [arch_model, infer_arch, model_name], [arch_model, infer_arch, model_name], show_progress="full")
    infer_arch.change(update_model_list, [arch_model, infer_arch], [model_name])
