import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task_type", default="sequence-llm", type=str, help='task type')
parser.add_argument("--infer_arch", default="Pytorch", choices=["Pytorch", "llama.cpp", "OpenVino", "ONNX"], type=str, help='infer arch')
parser.add_argument("--model_name", default="llama3", type=str, help='model name')
parser.add_argument("--model_version", default="Meta-Llama-3.1-8B-Instruct", type=str, help='model version')
