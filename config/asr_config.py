from typing import Dict, Any

MODEL_LIST: Dict[str, Any] = {
    "OpenAI_Whisper": {
        "model_provider_path": "models.audios.asr.model_openai_whisper",
        "model_provider_name": "OpenAIWhisperModel",
        "model_path": "/data/models/asr/whisper-large-v3"
    },
    "Ali_Paraformer": {
        "model_provider_path": "models.audios.asr.model_ali_paraformer",
        "model_provider_name": "AliParaformerModel",
        "model_path": "/data/models/asr/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    },
    "Meta_Wev2Vec_Conformer": {
        "model_provider_path": "models.audios.asr.model_meta_wev2vec_conformer",
        "model_provider_name": "MetaNLLBModel",
        "model_path": "/data/models/asr/wav2vec2-conformer-rope-large-960h-ft"
    }
}
