import os.path
import time
import torch
import torchaudio
from core.models.audios.tts.base_model import BaseModel, AUDIO_PATH
from chatterbox.mtl_tts import ChatterboxMultilingualTTS


class ChatterBoxModel(BaseModel):
    def load_model(self, model_path: str, device: str):
        model = ChatterboxMultilingualTTS.from_local(model_path, device=torch.device(device.lower()))

        self.device = device.lower()
        self.model = model

    def inference(self, texts: [], sample_wav: str = None):
        wavs = self.model.generate(
            texts,
            language_id="zh",
            audio_prompt_path=sample_wav,
        )
        audio_path = os.path.join(AUDIO_PATH, f"chatterbox_{int(time.time() * 1000)}")
        for i in range(len(wavs)):
            audio_path = f"{audio_path}_{i}.wav"
            try:
                torchaudio.save(audio_path, torch.from_numpy(wavs[i]).unsqueeze(0), self.model.sr)
            except:
                torchaudio.save(audio_path, torch.from_numpy(wavs[i]), self.model.sr)

        return audio_path

    def release(self):
        del self.model

