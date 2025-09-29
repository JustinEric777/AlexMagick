import os.path
import time
import torch
import torchaudio
from modules.models.audios.tts.base_model import BaseModel, AUDIO_PATH
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference


class VibeVoiceModel(BaseModel):
    def load_model(self, model_path: str, device: str):
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device.lower(),
            attn_implementation="sdpa",
        )

        self.device = device.lower()
        self.model = model

    def inference(self, texts: [], sample_wav: str = None):
        wavs = self.model.generate(
            texts,
            language_id="zh",
            audio_prompt_path=sample_wav,
        )
        audio_path = os.path.join(AUDIO_PATH, f"vibe_voice_{int(time.time() * 1000)}")
        for i in range(len(wavs)):
            audio_path = f"{audio_path}_{i}.wav"
            try:
                torchaudio.save(audio_path, torch.from_numpy(wavs[i]).unsqueeze(0), self.model.sr)
            except:
                torchaudio.save(audio_path, torch.from_numpy(wavs[i]), self.model.sr)

        return audio_path

    def release(self):
        del self.model
