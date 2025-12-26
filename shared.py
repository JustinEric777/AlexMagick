import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task_type", default="audio-tts", type=str, help='task type')
parser.add_argument("--infer_arch", default="Pytorch", choices=["Pytorch", "llama.cpp", "OpenVino", "ONNX"], type=str, help='infer arch')
parser.add_argument("--device", default="CPU", choices=["CPU", "GPU", "NPU", "cuda:0", "AUTO"], type=str, help='infer arch')
parser.add_argument("--model_name", default="ChatTTS", type=str, help='model name')
parser.add_argument("--model_version", default="ChatTTS", type=str, help='model version')
