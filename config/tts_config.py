from typing import Dict, Any

TASK_TYPE = "audio-tts"

MODEL_LIST: Dict[str, Any] = {
    "Pytorch": {
        "ChatTTS": {
            "model_provider_path": "modules.models.audios.tts.model_chat_tts",
            "model_provider_name": "ChatTTSModel",
            "model_path": "/data/models/tts",
            "model_list": [
                 "ChatTTS"
            ]
        },
        "CosyVoice-300M": {
            "model_provider_path": "modules.models.audios.tts.model_cosy_voice",
            "model_provider_name": "CosyVoiceModel",
            "model_path": "/data/models/tts",
            "model_list": [
                "CosyVoice-300M"
            ]
        },
        "Spark-TTS-0.5B": {
            "model_provider_path": "modules.models.audios.tts.model_spark_tts",
            "model_provider_name": "SparkTTSModel",
            "model_path": "/data/models/tts",
            "model_list": [
                "Spark-TTS-0.5B"
            ]
        },
        "fish-speech-1.5": {
            "model_provider_path": "modules.models.audios.tts.model_fish_tts",
            "model_provider_name": "FishTTSModel",
            "model_path": "/data/models/tts",
            "model_list": [
                "fish-speech-1.5"
            ]
        },
        "Zonos-v0.1-transformer": {
            "model_provider_path": "modules.models.audios.tts.model_zonos_tts",
            "model_provider_name": "ZonosTTSModel",
            "model_path": "/data/models/tts",
            "model_list": [
                "Zonos-v0.1-transformer"
            ]
        }
    }
}
