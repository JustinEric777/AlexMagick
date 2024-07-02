import gradio as gr
from shared import parser
from pages import sequence, image, audio


def get_default_tab_value(params: dict) -> dict:
    tabs = params["task_type"].split("-")
    params["default_first_tab"] = f"{tabs[0]}_tab"
    params["default_second_tab"] = f"{tabs[1]}_tab"

    return params


def create_ui(params):
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("""<h1 style="display:block; width:100%; margin: 20px"><center>Alex Magick AI Assistant</center></h1>""")

        with gr.Tabs(selected=params["default_first_tab"]):
            # create_ui
            sequence.create_ui(params)
            image.create_ui(params)
            audio.create_ui(params)

    demo.queue().launch(share=False, debug=True, server_name="0.0.0.0")


if __name__ == "__main__":
    args = parser.parse_args()
    args_dict = vars(args)
    create_ui(get_default_tab_value(args_dict))
