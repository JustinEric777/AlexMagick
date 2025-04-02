import torch
import torchaudio
import ChatTTS
from modules.models.audios.tts.base_model import BaseModel

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

        self.device = device
        self.model = model

    def inference(self, texts: [], audio_path: str, sample_wav: str):
        sample_audio = torchaudio.load(sample_wav)
        speaker = self.model.sample_audio_speaker(sample_audio)
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

        torchaudio.save("word_level_output.wav", torch.from_numpy(wavs[0]), 24000)

        return texts, audio_path

    def release(self):
        del self.model
