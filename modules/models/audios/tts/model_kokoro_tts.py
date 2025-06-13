import os
import time
import soundfile as sf
from kokoro import KPipeline
from kokoro import KModel
from modules.models.audios.tts.base_model import BaseModel, AUDIO_PATH

MODEL_NAMES = {
    'Kokoro-82M': 'kokoro-v1_0.pth',
    'Kokoro-82M-v1.1-zh': 'kokoro-v1_1-zh.pth',
}


class KokoroTTSModel(BaseModel):
    def __init__(self):
        self.voice_pt_path = None

    def load_model(self, model_path: str, device: str):
        config_path = os.path.join(model_path, "config.json")
        model_name = os.path.basename(model_path)
        model_file_path = os.path.join(model_path, MODEL_NAMES[model_name])
        k_model = KModel(
            model=model_file_path,
            config=config_path
        )

        model = KPipeline(
            model=k_model,
            lang_code='z',
            device=device.lower()
        )

        self.voice_pt_path = os.path.join(model_path, "voices")
        self.device = device.lower()
        self.model = model

    def inference(self, text: str, sample_wav: str = None):
        if "Kokoro-82M-v1.1-zh" in self.voice_pt_path:
            voice = 'af_maple'
        else:
            voice = 'af_heart'
        voice_pt = os.path.join(self.voice_pt_path, f"{voice}.pt")
        generator = self.model(
            text,
            voice=voice_pt,
            speed=1,
            split_pattern=r'\n+'
        )

        audio_path = os.path.join(AUDIO_PATH, f"fish_speech_{int(time.time() * 1000)}.wav")
        for i, (gs, ps, audio) in enumerate(generator):
            sf.write(audio_path, audio, 24000)

        return audio_path

    def release(self):
        del self.model