import gradio as gr

HOST_PREFIX = "http://10.172.10.75:7860/file="
GENERATE_TTS_AUDIO_PATH = "storage/tts"


def reload_model_ui(obj, args: dict):
    def reload_model(arch: str, device: str, model: str, version: str):
        return obj.reload_model(arch, device, model, version)

    def update_model_list(arch: str, model: str):
        model_list = obj.get_model_list(arch, model)
        return gr.Dropdown(
            label="model version",
            info="please choose model version",
            choices=model_list,
            value=model_list[0] if len(model_list) > 0 else "",
            interactive=True
        )

    def update_device(arch: str):
        device_list = obj.get_arch_device_list(arch)
        return gr.Dropdown(
            label="device",
            info="please choose model infer device",
            choices=device_list,
            value=device_list[0] if len(device_list) > 0 else "",
            interactive=True
        )

    def update_arch_model_list(arch: str):
        arch_model_list = obj.get_arch_model_list(arch)
        device_list = update_device(arch)
        model_list = update_model_list(arch, arch_model_list[0] if len(arch_model_list) > 0 else "")

        return device_list, gr.Dropdown(
            label="model name",
            info="please choose model name...",
            choices=arch_model_list,
            value=arch_model_list[0] if len(arch_model_list) > 0 else "",
            interactive=True
        ), model_list

    with gr.Row():
        with gr.Accordion("model inference setting", open=True):
            infer_arch = gr.Radio(
                label="infer arch",
                info="please choose a arch for inference",
                choices=obj.get_infer_arch_list(),
                value=args["infer_arch"] if args["infer_arch"] in obj.get_infer_arch_list() else obj.get_infer_arch_list()[0],
                interactive=True
            )
            device = gr.Dropdown(
                label="infer device",
                info="please choose model infer device",
                choices=obj.get_arch_device_list(infer_arch.value),
                value=args["device"] if args["device"] in obj.get_arch_device_list(infer_arch.value) else obj.get_arch_device_list(infer_arch.value)[0],
                interactive=True
            )
            model_name = gr.Dropdown(
                label="model name",
                info="please choose model name",
                choices=obj.get_arch_model_list(infer_arch.value),
                value=args["model_name"] if args["model_name"] in obj.get_arch_model_list(infer_arch.value) else obj.get_arch_model_list(infer_arch.value)[0],
                interactive=True
            )
            model_version = gr.Dropdown(
                label="model version",
                info="please choose model version",
                choices=obj.get_model_list(infer_arch.value, model_name.value),
                value=args["model_version"] if args["model_version"] in obj.get_model_list(infer_arch.value, model_name.value) else obj.get_model_list(infer_arch.value, model_name.value)[0],
                interactive=True
            )
    with gr.Row():
        model_reload = gr.Button("Reload Model ...", variant="primary")

    infer_arch.change(update_arch_model_list, [infer_arch], [device, model_name, model_version])
    model_name.change(update_model_list, [infer_arch, model_name], [model_version])
    model_reload.click(reload_model, [infer_arch, device, model_name, model_version], [infer_arch, device, model_name, model_version], show_progress="full")

    return infer_arch, device, model_name, model_version

