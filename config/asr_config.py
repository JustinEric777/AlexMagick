from typing import Dict, Any

TASK_TYPE = "asr"

MODEL_LIST: Dict[str, Any] = {
    "Pytorch": {
        "OpenAI_Whisper": {
            "model_provider_path": "core.models.audios.asr.model_openai_whisper",
            "model_provider_name": "OpenAIWhisperModel",
            "model_path": "/data/models/asr",
            "model_list": [
                "whisper-large-v3",
                "whisper-large-v3-turbo",
                "whisper-medium"
            ]
        },
        "Ali_FunASR": {
            "model_provider_path": "core.models.audios.asr.model_ali_funasr",
            "model_provider_name": "AliFunASRModel",
            "model_path": "/data/models/asr",
            "model_list": [
                "Fun-ASR-Nano-2512",
                "Fun-ASR-MLT-Nano-2512",
                "SenseVoiceSmall",
            ]
        },
        "Meta_Wev2Vec_Conformer": {
            "model_provider_path": "core.models.audios.asr.model_meta_wev2vec_conformer",
            "model_provider_name": "MetaWev2VecConformer",
            "model_path": "/data/models/asr",
            "model_list": [
                "wav2vec/wav2vec2-large-xlsr-53-english",
                "wav2vec/wav2vec2-large-xlsr-53-chinese-zh-cn",
            ]
        },
        "GLM-ASR": {
             "model_provider_path": "core.models.audios.asr.model_glm_asr",
             "model_provider_name": "GLMASRModel",
             "model_path": "/data/models/asr",
             "model_list": [
                 "GLM-ASR-Nano-2512"
             ]
        },
        "Kotoba-Whisper": {
             "model_provider_path": "core.models.audios.asr.model_kotoba_whisper",
             "model_provider_name": "KotobaWhisperModel",
             "model_path": "/data/models/asr",
             "model_list": [
                 "kotoba-whisper-v2.2"
             ]
        }
    },
    "OpenVino": {
        "Whisper": {
            "model_provider_path": "core.models.audios.asr.model_whisper_openvino",
            "model_provider_name": "WhisperOpenVINOModel",
            "model_path": "/data/models/asr",
            "model_list": [
                "whisper-large-v3-int4-ov",
                "whisper-large-v3-turbo-int8-ov"
            ]
        }
    }
}
