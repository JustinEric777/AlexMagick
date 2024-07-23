from typing import Dict, Any

TASK_TYPE = "sequence-mt"

MODEL_LIST: Dict[str, Any] = {
    "Pytorch": {
        "Ali_CSANMT": {
            "model_provider_path": "modules.models.sequences.mt.model_ali_csanmt",
            "model_provider_name": "AliCSANMTModel",
            "model_path": "/data/models/mt",
            "model_list": [
                "damo-csanmt-en-zh-large"
            ]
        },
        "Opus_mt_en_zh": {
            "model_provider_path": "modules.models.sequences.mt.model_opus_mt",
            "model_provider_name": "OpusMTModel",
            "model_path": "/data/models/mt",
            "model_list": [
                "opus-mt-en-zh"
            ]
        },
        "Meta_NLLB": {
            "model_provider_path": "modules.models.sequences.mt.model_meta_nllb",
            "model_provider_name": "MetaNLLBModel",
            "model_path": "/data/models/mt",
            "model_list": [
                "opus-mt-en-zh"
            ]
        },
        "Google_T5": {
            "model_provider_path": "modules.models.sequences.mt.model_google_t5",
            "model_provider_name": "GoogleT5Model",
            "model_path": "/data/models/mt",
            "model_list": [
                "t5_translate_en_ru_zh_large_1024"
            ]
        }
    }
}
