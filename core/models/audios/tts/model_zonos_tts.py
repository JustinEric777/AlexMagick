import torchaudio
import os
import time
from core.models.audios.tts.base_model import BaseModel, AUDIO_PATH
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict


class ZonosTTSModel(BaseModel):
    def load_model(self, model_path: str, device: str, **kwargs):
        # Handle cases where model_path points to directory or file
        if os.path.isdir(model_path):
            config_path = os.path.join(model_path, "config.json")
            weights_path = os.path.join(model_path, "model.safetensors")
        else:
            # Assume model_path is the weights file, guess config
            weights_path = model_path
            config_path = os.path.join(os.path.dirname(model_path), "config.json")
            
        model = Zonos.from_local(config_path, weights_path, device=device.lower())

        self.device = device
        self.model = model

    def inference(self, texts: str, sample_wav: str = None):
        if sample_wav is None:
            try:
                import zonos
                base_dir = os.path.dirname(zonos.__file__)
                candidate = os.path.join(base_dir, "examples", "exampleaudio.mp3")
                sample_wav = candidate if os.path.exists(candidate) else None
            except Exception:
                sample_wav = None
            if sample_wav is None:
                raise ValueError("Zonos 需要提供参考音频 sample_wav；请传入有效音频路径。")

        wav, sampling_rate = torchaudio.load(sample_wav)
        speaker = self.model.make_speaker_embedding(wav, sampling_rate)
        cond_dict = make_cond_dict(texts, speaker=speaker, language="cmn")
        conditioning = self.model.prepare_conditioning(cond_dict)

        codes = self.model.generate(conditioning)

        audio_path = os.path.join(AUDIO_PATH, f"zonos_tts_{int(time.time() * 1000)}.wav")
        wavs = self.model.autoencoder.decode(codes).cpu()
        torchaudio.save(audio_path, wavs[0], self.model.autoencoder.sampling_rate)

        return audio_path

    def release(self):
        del self.model
        del self.device

