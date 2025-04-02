from typing import Dict, Any

TASK_TYPE = "sequence-mt"

MODEL_LIST: Dict[str, Any] = {
    "Pytorch": {
        "Opus_mt_en_zh": {
            "model_provider_path": "modules.models.sequences.mt.model_opus_mt",
            "model_provider_name": "OpusMTModel",
            "model_path": "/data/models/mt",
            "model_list": [
                "opus-mt-en-zh",
                "opus-tatoeba-en-ja",
                "fugumt-en-ja",
            ]
        },
        "Meta_NLLB": {
            "model_provider_path": "modules.models.sequences.mt.model_meta_nllb",
            "model_provider_name": "MetaNLLBModel",
            "model_path": "/data/models/mt",
            "model_list": [
                "nllb-200-distilled-600M",
                "nllb-200-distilled-600M-en-zh_CN",
                "nllb-200-distilled-600M-ja-zh",
                "nllb-200-distilled-600M-ja-zh",
            ]
        },
        "Google_T5": {
            "model_provider_path": "modules.models.sequences.mt.model_google_t5",
            "model_provider_name": "GoogleT5Model",
            "model_path": "/data/models/mt",
            "model_list": [
                "t5_translate_en_ru_zh_large_1024",
                "t5_translate_en_ru_zh_large_1024_v2"
            ]
        },
        "Ali_CSANMT": {
            "model_provider_path": "modules.models.sequences.mt.model_ali_csanmt",
            "model_provider_name": "AliCSANMTModel",
            "model_path": "/data/models/mt",
            "model_list": [
                "nlp_csanmt_translation_en2zh_base",
                "damo-csanmt-en-zh-large",
            ]
        }
    }
}
