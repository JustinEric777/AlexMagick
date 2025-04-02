from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.models.vqgan.inference import load_model as load_decoder_model
from fish_speech.utils.schema import ServeTTSRequest
from modules.models.audios.tts.base_model import BaseModel


class FishTTSModel(BaseModel):
    def load_model(self, model_path: str, device: str):
        decoder_config_name = ""
        decoder_model = load_decoder_model(
            config_name=decoder_config_name,
            checkpoint_path=model_path,
            device=device,
        )

    def inference(self, text: str):
        pass

    def release(self):
        del self.model