import gradio as gr
from shared import parser
from pages import sequence, image, audio


def create_ui(params):
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("""<h1 style="display:block; width:100%; margin: 20px"><center>Alex Magick AI Assistant</center></h1>""")

        with gr.Tabs():
            # create_ui
            sequence.create_ui(params)
            image.create_ui(params)
            audio.create_ui(params)

    demo.queue().launch(share=False, debug=True, server_name="0.0.0.0")


if __name__ == "__main__":
    args = parser.parse_args()
    args_dict = vars(args)
    create_ui(args_dict)
