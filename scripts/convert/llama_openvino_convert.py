# optimum-cli export openvino --model /data/models/llama3/torch/Meta-Llama-3-8B --task text-generation-with-past --weight-format fp16 openvino/FP16
# optimum-cli export openvino --model /data/models/llama3/torch/Meta-Llama-3-8B --task text-generation-with-past --weight-format int8 openvino/INT8
import os.path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import openvino as ov

model_id = "/data/models/llm/pytorch/llama3/Meta-Llama-3.1-8B-Instruct"

output_dir = "/data/models/llama3/openvino"
fp16_model_dir = Path(output_dir) / "FP16"
int8_model_dir = Path(output_dir) / "INT8_compressed_weights"


def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 device_map="cpu",
                                                 torch_dtype=torch.float32,
                                                 low_cpu_mem_usage=True,
                                                 trust_remote_code=True)
    model.eval()

    return model, tokenizer


def convert_to_fp16():
    model, tokenizer = load_model(model_id)

    messages = [
        {"role": "system", "content": ""}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    ov_model = ov.convert_model(model, example_input=input_ids)
    ov.save_model(ov_model, os.path.join(fp16_model_dir, 'model.xml'))


def convert_to_int8():
    model, tokenizer = load_model(model_id)

    messages = [
        {"role": "system", "content": ""}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    ov_model = ov.convert_model(model, example_input=input_ids)
    ov.save_model(ov_model, os.path.join(fp16_model_dir, 'model.xml'))


if __name__ == "__main__":
    convert_to_fp16()

    convert_to_int8()
