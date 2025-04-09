import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task_type", default="multimodal-mllm", type=str, help='task type')
parser.add_argument("--infer_arch", default="Pytorch", choices=["Pytorch", "llama.cpp", "OpenVino", "ONNX"], type=str, help='infer arch')
parser.add_argument("--device", default="AUTO", choices=["CPU", "GPU", "NPU", "cuda:0", "AUTO"], type=str, help='infer arch')
parser.add_argument("--model_name", default="Qwen2.5-Omni", type=str, help='model name')
parser.add_argument("--model_version", default="Qwen2.5-Omni-7B", type=str, help='model version')
