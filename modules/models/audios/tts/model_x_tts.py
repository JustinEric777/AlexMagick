import os.path
import time
from modules.models.audios.tts.base_model import BaseModel, AUDIO_PATH
from TTS.api import TTS


class XTTSModel(BaseModel):
    def __init__(self):
        self.config = None

    def load_model(self, model_path: str, device: str):
        config_path =  os.path.join(model_path, "config.json")
        model = TTS(
            model_path=model_path,
            config_path=config_path,
            gpu=False,
            weights_only=True
        )

        self.model = model

    def inference(self, text: str, sample_wav: str = None):
        if sample_wav is None:
            sample_wav = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cosyvoice/examples/zero_shot_prompt.wav")
        else:
            sample_wav = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cosyvoice/examples/zero_shot_prompt.wav")

        audio_path = os.path.join(AUDIO_PATH, f"x_tts_{int(time.time() * 1000)}.wav")
        self.model.tts_to_file(
            text=text,
            file_path=audio_path,
            speaker_wav=sample_wav,
            language="zh-cn"
        )

        return audio_path

    def release(self):
        del self.model
