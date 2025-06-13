import os.path
import time
import torchaudio
from modules.models.audios.tts.base_model import BaseModel, AUDIO_PATH
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.cli.cosyvoice import CosyVoice


class CosyVoiceModel(BaseModel):
    def load_model(self, model_path: str, device: str):
        model = CosyVoice(model_path)

        self.device = device.lower()
        self.model = model

    def inference(self, texts: [], sample_wav: str = None):
        if sample_wav is None:
            sample_wav = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cosyvoice/examples/zero_shot_prompt.wav")
        else:
            sample_wav = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cosyvoice/examples/zero_shot_prompt.wav")

        prompt_speech_16k = load_wav(sample_wav, 16000)
        wavs = self.model.inference_zero_shot(
            texts,
            '希望你以后能够做的比我还好呦。',
            prompt_speech_16k=prompt_speech_16k,
            stream=False
        )

        audio_path = os.path.join(AUDIO_PATH, f"cosyvoice_{int(time.time() * 1000)}")
        for i, j in enumerate(wavs):
            audio_path = f"{audio_path}_{i}.wav"
            torchaudio.save(audio_path, j['tts_speech'], self.model.sample_rate)

        return audio_path

    def release(self):
        del self.model
