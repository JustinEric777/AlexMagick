import os.path
import time
import torch
import torchaudio
from modules.models.audios.tts.base_model import BaseModel, AUDIO_PATH
from voxcpm import VoxCPM


class VoxcpmModel(BaseModel):
    def load_model(self, model_path: str, device: str):
        model = VoxCPM.from_pretrained(model_path)

        self.device = device.lower()
        self.model = model

    def inference(self, texts: [], sample_wav: str = None):
        wavs = self.model.generate(
            text=texts,
            prompt_wav_path=sample_wav,      # optional: path to a prompt speech for voice cloning
            prompt_text=None,          # optional: reference text
            cfg_value=2.0,             # LM guidance on LocDiT, higher for better adherence to the prompt, but maybe worse
            inference_timesteps=10,   # LocDiT inference timesteps, higher for better result, lower for fast speed
            normalize=True,           # enable external TN tool
            denoise=True,             # enable external Denoise tool
            retry_badcase=True,        # enable retrying mode for some bad cases (unstoppable)
            retry_badcase_max_times=3,  # maximum retrying times
            retry_badcase_ratio_threshold=6.0
        )

        audio_path = os.path.join(AUDIO_PATH, f"voxcpm_{int(time.time() * 1000)}")
        for i in range(len(wavs)):
            audio_path = f"{audio_path}_{i}.wav"
            try:
                torchaudio.save(audio_path, torch.from_numpy(wavs[i]).unsqueeze(0), 16000)
            except:
                torchaudio.save(audio_path, torch.from_numpy(wavs[i]), 16000)

        return audio_path

    def release(self):
        del self.model
