import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task_type", default="sequence-llm", type=str, help='task type')
parser.add_argument("--infer_arch", default="Pytorch", choices=["Pytorch", "llama.cpp", "OpenVino", "ONNX"], type=str, help='infer arch')
parser.add_argument("--device", default="CPU", choices=["CPU", "GPU", "NPU", "AUTO"], type=str, help='infer arch')
parser.add_argument("--model_name", default="DeepSeek", type=str, help='model name')
parser.add_argument("--model_version", default="DeepSeek-R1-Distill-Qwen-7B", type=str, help='model version')
