from typing import Dict, Any

TASK_TYPE = "audio-tts"

MODEL_LIST: Dict[str, Any] = {
    "Pytorch": {
        "ChatTTS": {
            "model_provider_path": "core.models.audios.tts.model_chat_tts",
            "model_provider_name": "ChatTTSModel",
            "model_path": "/data/models/tts",
            "model_list": [
                 "ChatTTS"
            ]
        },
        "CosyVoice-300M": {
            "model_provider_path": "core.models.audios.tts.model_cosy_voice",
            "model_provider_name": "CosyVoiceModel",
            "model_path": "/data/models/tts",
            "model_list": [
                "CosyVoice-300M"
            ]
        },
        "Spark-TTS-0.5B": {
            "model_provider_path": "core.models.audios.tts.model_spark_tts",
            "model_provider_name": "SparkTTSModel",
            "model_path": "/data/models/tts",
            "model_list": [
                "Spark-TTS-0.5B"
            ]
        },
        "Fish-Speech-1.5": {
            "model_provider_path": "core.models.audios.tts.model_fish_tts",
            "model_provider_name": "FishTTSModel",
            "model_path": "/data/models/tts",
            "model_list": [
                "fish-speech-1.5"
            ]
        },
        "Kokoro-82M": {
            "model_provider_path": "core.models.audios.tts.model_kokoro_tts",
            "model_provider_name": "KokoroTTSModel",
            "model_path": "/data/models/tts",
            "model_list": [
                "Kokoro-82M",
                "Kokoro-82M-v1.1-zh",
            ]
        },
        "F5-TTS": {
            "model_provider_path": "core.models.audios.tts.model_f5_tts",
            "model_provider_name": "F5TTSModel",
            "model_path": "/data/models/tts",
            "model_list": [
                "F5-TTS"
            ]
        },
        "SpeechT5_TTS": {
            "model_provider_path": "core.models.audios.tts.model_speecht5_tts",
            "model_provider_name": "SpeechT5TTSModel",
            "model_path": "/data/models/tts",
            "model_list": [
                "speecht5_tts"
            ]
        },
        "Zonos-v0.1-transformer": {
            "model_provider_path": "core.models.audios.tts.model_zonos_tts",
            "model_provider_name": "ZonosTTSModel",
            "model_path": "/data/models/tts",
            "model_list": [
                "Zonos-v0.1-transformer"
            ]
        },
        "Dia-TTS": {
            "model_provider_path": "core.models.audios.tts.model_dia_tts",
            "model_provider_name": "DiaTTSModel",
            "model_path": "/data/models/tts",
            "model_list": [
                "Dia-1.6B"
            ]
        },
        "XTTS-v2": {
            "model_provider_path": "core.models.audios.tts.model_x_tts",
            "model_provider_name": "XTTSModel",
            "model_path": "/data/models/tts",
            "model_list": [
                "XTTS-v2"
            ]
        },
        "ChatterBox": {
            "model_provider_path": "core.models.audios.tts.model_chatterbox",
            "model_provider_name": "ChatterBoxModel",
            "model_path": "/data/models/tts",
            "model_list": [
                "chatterbox"
            ]
        },
        "Index-TTS": {
            "model_provider_path": "core.models.audios.tts.model_index_tts",
            "model_provider_name": "IndexTTSModel",
            "model_path": "/data/models/tts",
            "model_list": [
                "IndexTTS-2"
            ]
        },
        "Oute-TTS": {
            "model_provider_path": "core.models.audios.tts.model_oute_tts",
            "model_provider_name": "OuteTTSModel",
            "model_path": "/data/models/tts",
            "model_list": [
                "OuteTTS-1.0-0.6B"
            ]
        },
        "Vibe-Voice": {
            "model_provider_path": "core.models.audios.tts.model_vibe_voice",
            "model_provider_name": "VibeVoiceModel",
            "model_path": "/data/models/tts",
            "model_list": [
                "VibeVoice-1.5B"
            ]
        },
        "Voxcpm": {
            "model_provider_path": "core.models.audios.tts.model_voxcpm",
            "model_provider_name": "VoxcpmModel",
            "model_path": "/data/models/tts",
            "model_list": [
                "VoxCPM-0.5B"
            ]
        }
    }
}
