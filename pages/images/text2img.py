import gradio as gr
import numpy as np
import random
from pages.common import reload_model_ui, HOST_PREFIX
from modules import text2img

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024


def generate(positive_prompt, negative_prompt, randomize_seed, seed, guidance_scale, num_inference_steps, width, height):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    result = text2img.generate(positive_prompt, negative_prompt, seed, guidance_scale, num_inference_steps, width, height)
    seed_result = (*result, seed)

    return seed_result


def create_ui(args: dict):
    text2img.init_model(args)

    with gr.Tab(label="Text2Image Model", id="image_text2image_tab") as image_text2img_tab:
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Row():
                    with gr.Column(scale=2):
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

                with gr.Row():
                    with gr.Column(scale=4):
                        gr.Examples(
                            label="Input Examples",
                            examples=[
                                ["realistic, 1girl, (detail skin texture, ultra-detailed body), atmospheric scene, masterpiece, best quality,(cinematic light), solo, midriff, smile, navel, white crop top, brown hair, shirt, grin, outdoors, standing, [[pink headband]],[sport shorts]"],
                                ["The image features a beautiful young woman wearing a white shirt and a black hat. She is standing in a grassy area, possibly a park, and appears to be posing for a picture. The woman is wearing a necklace, adding a touch of elegance to her outfit. The scene is bright and sunny, creating a pleasant atmosphere for the photo. High fashion,  dynamic, dramatic, elegant, high fashion Vogue cover Dramatic photography, supermodel, avant garde style"],
                                ["The most beautiful woman ever seen wearing an off-shoulder tank top and loose shorts, lying back on the floor next to a fireplace. Beautiful sunset afternoon lights coming from the large window. (best quality, masterpiece:1.2), photorealistic, intricate details, symmetrical eyes, (beautiful detailed face), detailed lips, absurdres, dynamic angle, eyeliner, dark hair, yellowish eyes, slim waist, narrow shoulders."],
                                ["A 35 yo latin woman in a delightful afternoon. This portrait is the best representation of female beauty, shiny dark hair, hazel eyes, perfect olive skin. Extremely realistic textures and warm colors give the final touch. Sharp focus and realistic shadows add to the scene."]
                            ],
                            elem_id="text2image_examples",
                            inputs=[positive_prompt],
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
                    infer_arch, device, model_name, model_version = reload_model_ui(text2img, args)
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

        def update_results(positive_prompt_value, negative_prompt_value, result_image_value, metric_value):
            items = results.value["data"]
            image = f"""<img src="{HOST_PREFIX}{result_image_value}" style="width: 100px, height: auto" ></img>"""
            new_row = [positive_prompt_value, negative_prompt_value, image, metric_value.strip()]
            if len(items[0][0]) == 0 and len(items[0][1]) == 0:
                items[0] = new_row
            else:
                items.append(new_row)
            return items

        generate_bt.click(generate, inputs=[positive_prompt, negative_prompt, randomize_seed, seed, guidance_scale, num_inference_steps, width, height], outputs=[image_output, metric, seed], queue=False).then(
            update_results, inputs=[positive_prompt, negative_prompt, image_output, metric], outputs=[results], queue=False
        )
        clear.click(lambda: None, None, [positive_prompt, positive_prompt, image_output, results], queue=False)

    image_text2img_tab.select(text2img.reload_model, [infer_arch, device, model_name, model_version], [infer_arch, device, model_name, model_version])

