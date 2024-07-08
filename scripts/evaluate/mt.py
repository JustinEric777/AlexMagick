import importlib
import json
import time
import os
import csv
from datasets import load_dataset
from nltk.translate.bleu_score import corpus_bleu
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../../')))
from config.mt_config import MODEL_LIST
from modules.mt.model_opus_mt import OpusMTModel
from modules.mt.model_ali_csanmt import AliCSANMTModel
from modules.mt.model_meta_mbart import MetaMBartModel
from modules.mt.model_meta_m2m import MetaM2MModel
from modules.mt.model_meta_nllb import MetaNLLBModel
from modules.mt.model_google_t5 import GoogleT5Model

def load_data():
    data_files = {
        "en_zh": "../datasets/srt/result.json",
    }
    dataset = load_dataset("json", data_files=data_files)

    return dataset["en_zh"]


def load_models():
    models = {}
    for key, value in MODEL_LIST.items():
        model_class = value["model_provider_name"]
        model_path = value["model_path"]
        model_provider_path = f'modules.{MODEL_LIST[key]["model_provider_path"]}'
        mt_class_name = getattr(importlib.import_module(model_provider_path), model_class)
        mt_object = mt_class_name()
        mt_object.load_model(model_path)
        models[key] = mt_object

    return models


if __name__ == "__main__":
    datas = load_data()
    models = load_models()

    result_file_path = "../datasets/srt/result.csv"
    with open(result_file_path, 'w+', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        header = [
            "model_name",
            "original_en_text",
            "original_zh_text",
            "translated_text",
            "score",
            "cost_time",
            "words_count",
            "single_word_cost_time"
        ]
        writer.writerow(header)

        for data in datas:
            line_result = {}
            original_en_result = data["en"]
            original_zh_result = data["zh"]

            for key, model in models.items():
                start_time = time.time()

                translated_text = model.translate(original_en_result)
                score = corpus_bleu([[original_zh_result]], [translated_text], weights=(0.5, 0.4, 0.1, 0))

                original_en_result = original_en_result.replace("-", "").strip()
                original_zh_result = original_zh_result.replace("-", "").strip()
                translated_text = translated_text.replace("-", "").strip()
                generated_result = {
                    "model_name": key,
                    "original_en_text": original_en_result,
                    "original_zh_text": original_zh_result,
                    "translated_text": translated_text,
                    "score": score,
                    "cost_time": round(time.time()-start_time, 3),
                    "words_count": len(translated_text),
                    "single_word_cost_time": 0 if len(translated_text) == 0 else round((time.time()-start_time)/len(translated_text), 3)
                }
                print(json.dumps(generated_result, ensure_ascii=False))
                writer.writerow(generated_result.values())







