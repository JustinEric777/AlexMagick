import gradio as gr
import numpy as np
import random
from pages.common import reload_model_ui, HOST_PREFIX
from modules import inpainting

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024


def generate(image_input, positive_prompt, negative_prompt, randomize_seed, seed, guidance_scale, num_inference_steps, width, height):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    result = inpainting.generate(image_input, positive_prompt, negative_prompt, seed, guidance_scale, num_inference_steps, width, height)
    seed_result = (*result, seed)

    return seed_result


def create_ui(args: dict):
    inpainting.init_model(args)

    with gr.Tab(label="Image2Image Model", id="image_image2image_tab") as image_inpainting_tab:
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    image_base = gr.Image(type="filepath", label="Base Image", sources=["upload"])
                with gr.Row():
                    image_mask = gr.Image(type="filepath", label="Mask Image", sources=["upload"])
                with gr.Row():
                    positive_prompt = gr.Textbox(label="Positive Prompt", lines=6, placeholder="Positive Prompt Text...",)
                with gr.Row():
                    negative_prompt = gr.Textbox(label="Negative Prompt", lines=6, placeholder="Negative Prompt Text...",)
                with gr.Row():
                    generate_bt = gr.Button("Generate", variant="primary")
                    clear = gr.Button("Clear")
                    metric = gr.Textbox(visible=False)
            with gr.Column(scale=2):
                with gr.Row():
                    image_output = gr.Image(type="filepath", label="Generated Output Image", interactive=False)
            with gr.Column(scale=1):
                infer_arch, model_name, model_version = reload_model_ui(inpainting, args)

        with gr.Row():
            with gr.Column(scale=4):
                gr.Examples(
                    label="Positive Prompt Examples",
                    examples=[
                        [
                            """<img src="https://www.wehelpwin.com/Editor/ewebeditor/uploadfile/20231116104054532002.jpg" />""",
                            """<img src="https://www.wehelpwin.com/Editor/ewebeditor/uploadfile/20231116104054532002.jpg" />""",
                            "a woman with a short hair and a white shirt is posing for a picture with her hand on her chin, a photorealistic painting, Ayami Kojima, precisionism, perfect face",
                            "dongwm-nt,bad finger, bad body"
                        ],
                    ],
                    elem_id="text2image_examples",
                    inputs=[image_base, image_mask, positive_prompt, negative_prompt],
                )

                results = gr.Dataframe(
                    label="Image Generated Results",
                    headers=["Positive Prompt", "Negative Prompt", "result", "Metric"],
                    datatype="markdown",
                    column_widths=[30, 30, 20, 20],
                    wrap=True
                )
            with gr.Column(scale=1):
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
                                value=20,
                            )

        def update_results(image_input_value, image_mask_value, positive_prompt_value, negative_prompt_value, result_image_value, metric_value):
            items = results.value["data"]
            input_image = f"""<img src="{HOST_PREFIX}{image_input_value}" style="width: 100px, height: auto" ></img>"""
            mask_image = f"""<img src="{HOST_PREFIX}{image_mask_value}" style="width: 100px, height: auto" ></img>"""
            output_image = f"""<img src="{HOST_PREFIX}{result_image_value}" style="width: 100px, height: auto" ></img>"""
            new_row = [input_image, mask_image, positive_prompt_value, negative_prompt_value, output_image, metric_value.strip()]
            if len(items[0][0]) == 0 and len(items[0][1]) == 0:
                items[0] = new_row
            else:
                items.append(new_row)
            return items

        generate_bt.click(generate, inputs=[image_base, image_mask, positive_prompt, negative_prompt, randomize_seed, seed, guidance_scale, num_inference_steps, width, height], outputs=[image_output, metric, seed], queue=False).then(
            update_results, inputs=[image_base, image_mask, positive_prompt, negative_prompt, image_output, metric], outputs=[results], queue=False
        )
        clear.click(lambda: None, None, [positive_prompt, positive_prompt, image_output, results], queue=False)

    image_inpainting_tab.select(inpainting.reload_model, [infer_arch, model_name, model_version], [infer_arch, model_name, model_version])

