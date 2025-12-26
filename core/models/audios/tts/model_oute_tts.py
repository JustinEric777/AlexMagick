import os.path
import time
import torch
import torchaudio
import outetts
from core.models.audios.tts.base_model import BaseModel, AUDIO_PATH


class OuteTTSModel(BaseModel):
    def load_model(self, model_path: str, device: str):
        model = outetts.Interface(
            config=outetts.ModelConfig.auto_config(
                model=outetts.Models.VERSION_1_0_SIZE_1B,
                backend=outetts.Backend.LLAMACPP,
                quantization=outetts.LlamaCppQuantization.FP16
            )
        )

        self.device = device.lower()
        self.model = model

    def inference(self, texts: [], sample_wav: str = None):
        speaker = self.model.load_default_speaker("EN-FEMALE-1-NEUTRAL")

        wavs = self.model.generate(
            config=outetts.GenerationConfig(
                text=texts,
                speaker=speaker,
            )
        )

        audio_path = os.path.join(AUDIO_PATH, f"oute_tts_{int(time.time() * 1000)}")
        for i in range(len(wavs)):
            audio_path = f"{audio_path}_{i}.wav"
            try:
                torchaudio.save(audio_path, torch.from_numpy(wavs[i]).unsqueeze(0), self.model.sr)
            except:
                torchaudio.save(audio_path, torch.from_numpy(wavs[i]), self.model.sr)

        return audio_path

    def release(self):
        del self.model

