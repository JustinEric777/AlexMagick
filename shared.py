import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--arch_model", default="llama3", type=str, help='mode name')
parser.add_argument("--infer_arch", default="Transformers", type=str, help='arch name')
parser.add_argument("--model_name", default="OpenAI_Whisper", type=str, help='mode name')
parser.add_argument("--task_type", default="audio-asr", type=str, help='task type')
parser.add_argument("--is_4bit", action='store_true', help='use 4bit model')
