import gradio as gr
import numpy as np
import random
from pages.common import create_standard_page, create_image_html, HOST_PREFIX
from servers import img2img_server

img2img = img2img_server.Img2ImgServer()

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024


def generate(image_input, positive_prompt, negative_prompt, randomize_seed, seed, guidance_scale, num_inference_steps, width, height):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    result = img2img.generate(image_input, positive_prompt, negative_prompt, seed, guidance_scale, num_inference_steps, width, height)
    seed_result = (*result, seed)

    return seed_result


def create_ui(args: dict):
    components = {}

    def render_main():
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    components['positive_prompt'] = gr.Textbox(label="Positive Prompt", lines=6, placeholder="Positive Prompt Text...",)
                with gr.Row():
                    components['image_input'] = gr.Image(type="filepath", label="Base Image", sources=["upload"])
                with gr.Row():
                    components['generate_bt'] = gr.Button("Generate", variant="primary")
                    components['clear'] = gr.Button("Clear")
                    components['metric'] = gr.Textbox(visible=False)
            with gr.Column(scale=2):
                with gr.Row():
                    components['negative_prompt'] = gr.Textbox(label="Negative Prompt", lines=6, placeholder="Negative Prompt Text...",)
                with gr.Row():
                    components['image_output'] = gr.Image(type="filepath", label="Generated Output Image", interactive=False)

        with gr.Row():
            with gr.Column(scale=4):
                gr.Examples(
                    label="Input Examples",
                    examples=[
                        [
                            "./pages/examples/images/img2img_1.jpg",
                            "a woman with a short hair and a white shirt is posing for a picture with her hand on her chin, a photorealistic painting, Ayami Kojima, precisionism, perfect face",
                            "dongwm-nt,bad finger, bad body"
                        ],
                        [
                            "./pages/examples/images/img2img_2.png",
                            "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k",
                            ""
                        ],
                        [
                            "./pages/examples/images/img2img_3.png",
                            "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
                            ""
                        ],
                    ],
                    elem_id="text2image_examples",
                    inputs=[components['image_input'], components['positive_prompt'], components['negative_prompt']],
                )

                components['results'] = gr.Dataframe(
                    label="Image Generated Results",
                    headers=["Init Image",  "Positive Prompt", "Negative Prompt", "result", "Metric"],
                    datatype="markdown",
                    column_widths=[15, 25, 25, 15, 20],
                    wrap=True
                )

    def render_params():
        with gr.Accordion("model params", open=True):
            components['seed'] = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
            components['randomize_seed'] = gr.Checkbox(label="Randomize seed", value=True)
            components['width'] = gr.Slider(label="Width", minimum=256, maximum=MAX_IMAGE_SIZE, step=64, value=1024)
            components['height'] = gr.Slider(label="Height", minimum=256, maximum=MAX_IMAGE_SIZE, step=64, value=1024)
            components['guidance_scale'] = gr.Slider(label="Guidance scale", minimum=0.0, maximum=10.0, step=0.1, value=5.0)
            components['num_inference_steps'] = gr.Slider(label="Number of inference steps", minimum=1, maximum=50, step=1, value=20)

    create_standard_page(img2img, args, render_main, render_params_ui=render_params, tab_label="Image2Image Model", tab_id="image_img2img_tab")

    def update_results(image_input_value, positive_prompt_value, negative_prompt_value, result_image_value, metric_value):
        items = components['results'].value["data"]
        input_image = create_image_html(image_input_value)
        output_image = create_image_html(result_image_value)
        new_row = [input_image, positive_prompt_value, negative_prompt_value, output_image, metric_value.strip()]
        items.append(new_row)
        return items

    components['generate_bt'].click(
        generate, 
        inputs=[
            components['image_input'], components['positive_prompt'], components['negative_prompt'], 
            components['randomize_seed'], components['seed'], 
            components['guidance_scale'], components['num_inference_steps'], 
            components['width'], components['height']
        ], 
        outputs=[components['image_output'], components['metric'], components['seed']], 
        queue=False
    ).then(
        update_results, 
        inputs=[components['image_input'], components['positive_prompt'], components['negative_prompt'], components['image_output'], components['metric']], 
        outputs=[components['results']], 
        queue=False
    )
    
    components['clear'].click(
        lambda: None, 
        None, 
        [components['positive_prompt'], components['positive_prompt'], components['image_output'], components['results']], 
        queue=False
    )
