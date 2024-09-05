import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task_type", default="sequence-text2embedding", type=str, help='task type')
parser.add_argument("--infer_arch", default="Pytorch", choices=["Pytorch", "llama.cpp", "OpenVino", "ONNX"], type=str, help='infer arch')
parser.add_argument("--model_name", default="transformer_sentence", type=str, help='model name')
parser.add_argument("--model_version", default="bge-large-zh-v1.5", type=str, help='model version')
