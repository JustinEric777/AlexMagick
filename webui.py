import os
import gradio as gr
from shared import parser
from pages import sequence, image, audio, video, multimodal


def gradio_default_setting():
    # close analytics_enabled for gradio
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    # tmp dir
    os.environ["GRADIO_TEMP_DIR"] = "storage"
    # allow access paths
    os.environ["GRADIO_ALLOWED_PATHS"] = "storage"
    # examples cache
    os.environ["GRADIO_CACHE_EXAMPLES"] = "False"
    # example cache dir
    os.environ["GRADIO_EXAMPLES_CACHE"] = "pages/examples/"
    # cache mode
    # os.environ["GRADIO_CACHE_MODE"] = "False"


custom_css = """
<style>
.text2image_examples {
    white-space: break-space;
    text-align: left;
    padding: 10px;
}
</style>
"""


def get_default_tab_value(params: dict) -> dict:
    tabs = params["task_type"].split("-")
    params["default_first_tab"] = f"{tabs[0]}_tab"
    params["default_second_tab"] = f"{tabs[1]}_tab"

    return params


def create_ui(params: dict):
    # default settings
    gradio_default_setting()

    with gr.Blocks(
        theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"]),
        css=custom_css,
    ) as demo:
        gr.Markdown("""<h1 style="display:block; width:100%; margin: 20px; font-size: 32px"><center>Alex Magick AI Assistant</center></h1>""")

        with gr.Tabs(selected=params["default_first_tab"]):
            # create_ui
            sequence.create_ui(params)
            image.create_ui(params)
            audio.create_ui(params)
            video.create_ui(params)
            multimodal.create_ui(params)

    demo.queue().launch(share=False, debug=True, server_name="0.0.0.0")


if __name__ == "__main__":
    args = parser.parse_args()
    args_dict = vars(args)

    # set default tab
    args = get_default_tab_value(args_dict)
    create_ui(args)
