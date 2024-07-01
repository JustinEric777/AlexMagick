# optimum-cli export openvino --model /data/models/llama3/torch/Meta-Llama-3-8B --task text-generation-with-past --weight-format fp16 openvino/FP16
# optimum-cli export openvino --model /data/models/llama3/torch/Meta-Llama-3-8B --task text-generation-with-past --weight-format int8 openvino/INT8
import os
from llm_config import SUPPORTED_LLM_MODELS
from pathlib import Path

output_dir = "/data/models/llama3/openvino"
model_language = "English"
model_id = "llama-3-8b-instruct"
model_configuration = SUPPORTED_LLM_MODELS[model_language][model_id]

pt_model_id = "/data/models/llama3/torch/Meta-Llama-3-8B"
fp16_model_dir = Path(output_dir) / "FP16"
int8_model_dir = Path(output_dir) / "INT8_compressed_weights"
int4_model_dir = Path(output_dir) / "INT4_compressed_weights"


def convert_to_fp16():
    if (fp16_model_dir / "openvino_model.xml").exists():
        print("model existed...")
        return
    remote_code = model_configuration.get("remote_code", False)
    export_command_base = "optimum-cli export openvino --model {} --task text-generation-with-past --weight-format fp16".format(pt_model_id)
    if remote_code:
        export_command_base += " --trust-remote-code"
    export_command = export_command_base + " " + str(fp16_model_dir)
    print("export_command : ", export_command)
    os.system(export_command)


def convert_to_int8():
    if (int8_model_dir / "openvino_model.xml").exists():
        print("model existed...")
        return
    int8_model_dir.mkdir(parents=True, exist_ok=True)
    remote_code = model_configuration.get("remote_code", False)
    export_command_base = "optimum-cli export openvino --model {} --task text-generation-with-past --weight-format int8".format(pt_model_id)
    if remote_code:
        export_command_base += " --trust-remote-code"
    export_command = export_command_base + " " + str(int8_model_dir)
    print("export_command : ", export_command)
    os.system(export_command)


def convert_to_int4():
    compression_configs = {
        "zephyr-7b-beta": {
            "sym": True,
            "group_size": 64,
            "ratio": 0.6,
        },
        "mistral-7b": {
            "sym": True,
            "group_size": 64,
            "ratio": 0.6,
        },
        "minicpm-2b-dpo": {
            "sym": True,
            "group_size": 64,
            "ratio": 0.6,
        },
        "gemma-2b-it": {
            "sym": True,
            "group_size": 64,
            "ratio": 0.6,
        },
        "notus-7b-v1": {
            "sym": True,
            "group_size": 64,
            "ratio": 0.6,
        },
        "neural-chat-7b-v3-1": {
            "sym": True,
            "group_size": 64,
            "ratio": 0.6,
        },
        "llama-2-chat-7b": {
            "sym": True,
            "group_size": 128,
            "ratio": 0.8,
        },
        "llama-3-8b-instruct": {
            "sym": True,
            "group_size": 128,
            "ratio": 0.8,
        },
        "gemma-7b-it": {
            "sym": True,
            "group_size": 128,
            "ratio": 0.8,
        },
        "chatglm2-6b": {
            "sym": True,
            "group_size": 128,
            "ratio": 0.72,
        },
        "qwen-7b-chat": {"sym": True, "group_size": 128, "ratio": 0.6},
        "red-pajama-3b-chat": {
            "sym": False,
            "group_size": 128,
            "ratio": 0.5,
        },
        "default": {
            "sym": False,
            "group_size": 128,
            "ratio": 0.8,
        },
    }

    model_compression_params = compression_configs.get(model_id, compression_configs["default"])
    if (int4_model_dir / "openvino_model.xml").exists():
        print("model existed...")
        return
    remote_code = model_configuration.get("remote_code", False)
    export_command_base = "optimum-cli export openvino --model {} --task text-generation-with-past --weight-format int4".format(pt_model_id)
    int4_compression_args = " --group-size {} --ratio {}".format(model_compression_params["group_size"], model_compression_params["ratio"])
    if model_compression_params["sym"]:
        int4_compression_args += " --sym"
    export_command_base += int4_compression_args
    if remote_code:
        export_command_base += " --trust-remote-code"
    export_command = export_command_base + " " + str(int4_model_dir)
    print("export_command : ", export_command)
    os.system(export_command)


if __name__ == "__main__":
    # convert_to_fp16()

    convert_to_int8()

    convert_to_int4()