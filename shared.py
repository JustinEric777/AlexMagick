import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task_type", default="image-text2img", type=str, help='task type')
parser.add_argument("--infer_arch", default="Pytorch", choices=["Pytorch", "llama.cpp", "OpenVino", "IPEX-LLM"], type=str, help='infer arch')
parser.add_argument("--model_name", default="SD-1.5", type=str, help='model name')
parser.add_argument("--model_version", default="stable-diffusion-v1-5", type=str, help='model version')
