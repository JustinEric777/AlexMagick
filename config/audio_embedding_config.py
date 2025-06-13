from typing import Dict, Any

TASK_TYPE = "audio-audio2embedding"

MODEL_LIST: Dict[str, Any] = {
    "Pytorch": {
        "clap": {
            "model_provider_path": "modules.models.audios.retrieval.model_laion_clap",
            "model_provider_name": "LaionClapModel",
            "model_path": "/data/models/audio_clip",
            "model_list": [
                "larger_clap_music_and_speech",
                "larger_clap_general",
                "clap-htsat-fused",
                "clap-htsat-unfused",
                "clap-htsat-fused-ko",
            ]
        },
        "msclap": {
            "model_provider_path": "modules.models.audios.retrieval.model_microsoft_clap",
            "model_provider_name": "MicrosoftClapModel",
            "model_path": "/data/models/audio_clip",
            "model_list": [
                "msclap/CLAP_weights_2023.pth",
            ]
        },
        "clap_ipa": {
            "model_provider_path": "modules.models.audios.retrieval.model_anyspeech_clap_ipa",
            "model_provider_name": "AnySpeechClapIpaModel",
            "model_path": "/data/models/audio_clip",
            "model_list": [
                "clap-ipa-base",
                "clap-ipa-small",
                "clap-ipa-tiny",
            ]
        },
    }
}
