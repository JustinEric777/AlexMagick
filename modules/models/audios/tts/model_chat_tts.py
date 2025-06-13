import os.path
import time
import torch
import torchaudio
import ChatTTS
from modules.models.audios.tts.base_model import BaseModel, AUDIO_PATH

torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')


class ChatTTSModel(BaseModel):
    def load_model(self, model_path: str, device: str):
        model = ChatTTS.Chat()
        model.load(
            source="local",
            force_redownload=False,
            custom_path=model_path
        )

        self.device = device.lower()
        self.model = model

    def inference(self, texts: [], sample_wav: str = None):
        # sample_audio = torchaudio.load(sample_wav)
        # speaker = self.model.sample_audio_speaker(sample_audio)
        speaker =  self.model.sample_random_speaker()
        params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_emb=speaker,
            temperature=0.3,
        )

        params_refine_text = ChatTTS.Chat.RefineTextParams(
            prompt='[oral_2][laugh_0][break_6]',
            top_P=0.7,
            top_K=20,
        )

        wavs = self.model.infer(
            texts,
            params_refine_text=params_refine_text,
            params_infer_code=params_infer_code,
        )

        audio_path = os.path.join(AUDIO_PATH, f"chattts_{int(time.time() * 1000)}")
        for i in range(len(wavs)):
            audio_path = f"{audio_path}_{i}.wav"
            try:
                torchaudio.save(audio_path, torch.from_numpy(wavs[i]).unsqueeze(0), 24000)
            except:
                torchaudio.save(audio_path, torch.from_numpy(wavs[i]), 24000)

        return audio_path

    def release(self):
        del self.model
