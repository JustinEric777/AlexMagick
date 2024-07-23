from typing import Dict, Any

TASK_TYPE = "audio-asr"

MODEL_LIST: Dict[str, Any] = {
    "Pytorch": {
        "OpenAI_Whisper": {
            "model_provider_path": "modules.models.audios.asr.model_openai_whisper",
            "model_provider_name": "OpenAIWhisperModel",
            "model_path": "/data/models/asr",
            "model_list": [
                "whisper-large-v3"
            ]
        },
        "Ali_Paraformer": {
            "model_provider_path": "modules.models.audios.asr.model_ali_paraformer",
            "model_provider_name": "AliParaformerModel",
            "model_path": "/data/models/asr",
            "model_list": [
                "speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
            ]
        },
        "Meta_Wev2Vec_Conformer": {
            "model_provider_path": "modules.models.audios.asr.model_meta_wev2vec_conformer",
            "model_provider_name": "MetaNLLBModel",
            "model_path": "/data/models/asr",
            "model_list": [
                "wav2vec2-conformer-rope-large-960h-ft"
            ]
        }
    }
}
