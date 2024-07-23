import gradio as gr
import numpy as np
from pages.common import reload_model_ui
from modules import img2img

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1344


def create_ui(args: dict):
    with gr.Tab(label="Image2Image Model", id="image_image2image_tab") as image_img2img_tab:
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    positive_prompt = gr.Textbox(label="Positive Prompt", lines=5, placeholder="Positive Prompt Text...",)
                with gr.Row():
                    negative_prompt = gr.Textbox(label="Negative Prompt", lines=5, placeholder="Negative Prompt Text...",)
                with gr.Row():
                    generate_bt = gr.Button("Generate", variant="primary")
                    clear = gr.Button("Clear")
                    metric = gr.Textbox(visible=False)
                with gr.Row():
                    results = gr.Dataframe(
                        label="Image Generated Results",
                        headers=["Positive Prompt", "Negative Prompt", "Metric"],
                        datatype="markdown",
                        column_widths=[40, 40, 20],
                        wrap=True
                    )
            with gr.Column(scale=2):
                with gr.Row():
                    image_output = gr.Image(label="Generated Output Image", height=380, interactive=False)
            with gr.Column(scale=1):
                infer_arch, model_name, model_version = reload_model_ui(img2img, args)

                with gr.Row():
                    with gr.Accordion("model params", open=True):
                        seed = gr.Slider(
                            label="Seed",
                            minimum=0,
                            maximum=MAX_SEED,
                            step=1,
                            value=0,
                        )
                        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                        with gr.Row():
                            width = gr.Slider(
                                label="Width",
                                minimum=256,
                                maximum=MAX_IMAGE_SIZE,
                                step=64,
                                value=1024,
                            )
                            height = gr.Slider(
                                label="Height",
                                minimum=256,
                                maximum=MAX_IMAGE_SIZE,
                                step=64,
                                value=1024,
                            )
                        with gr.Row():
                            guidance_scale = gr.Slider(
                                label="Guidance scale",
                                minimum=0.0,
                                maximum=10.0,
                                step=0.1,
                                value=5.0,
                            )
                            num_inference_steps = gr.Slider(
                                label="Number of inference steps",
                                minimum=1,
                                maximum=50,
                                step=1,
                                value=28,
                            )

        def update_results(original_audio, translated_text, metric_value):
            items = results.value["data"]
            new_row = [original_audio, translated_text, metric_value.strip()]
            if len(items[0][0]) == 0 and len(items[0][1]) == 0:
                items[0] = new_row
            else:
                items.append(new_row)
            return items

        generate_bt.click(img2img.generate, inputs=[positive_prompt, negative_prompt, seed, guidance_scale, num_inference_steps, width, height, model_name], outputs=[image_output, metric], queue=False).then(
            update_results, inputs=[positive_prompt, positive_prompt, metric], outputs=[results], queue=False
        )
        clear.click(lambda: None, None, [positive_prompt, positive_prompt, results], queue=False)

    image_img2img_tab.select(img2img.reload_model, [infer_arch, model_name, model_version], [infer_arch, model_name, model_version])
