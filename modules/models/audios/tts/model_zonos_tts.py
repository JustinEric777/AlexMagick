import torch
import torchaudio
import os
import time
from modules.models.audios.tts.base_model import BaseModel, AUDIO_PATH
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict


class ZonosTTSModel(BaseModel):
    def load_model(self, model_path: str, device: str):
        config_path = os.path.join(model_path, "config.json")
        model_path = os.path.join(model_path, "model.safetensors")
        model = Zonos.from_local(config_path, model_path, device=device.lower())

        self.device = device
        self.model = model

    def inference(self, texts: str, sample_wav: str = None):
        if sample_wav is None:
            sample_wav = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zonos/examples/exampleaudio.mp3")
        else:
            sample_wav = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zonos/examples/exampleaudio.mp3")

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
