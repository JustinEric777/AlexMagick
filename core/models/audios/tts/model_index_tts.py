import os.path
import time
import torch
import torchaudio
from core.models.audios.tts.base_model import BaseModel, AUDIO_PATH
from indextts.infer_v2 import IndexTTS2


class IndexTTSModel(BaseModel):
    def load_model(self, model_path: str, device: str):
        model = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)

        self.device = device.lower()
        self.model = model

    def inference(self, texts: [], sample_wav: str = None):
        audio_path = os.path.join(AUDIO_PATH, f"index_tts_{int(time.time() * 1000)}")
        self.model.generate(
            text=texts,
            output_path=audio_path,
            spk_audio_prompt=sample_wav,
            mo_alpha=0.9,
            verbose=True
        )

        return audio_path

    def release(self):
        del self.model

