import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from modules.models.audios.tts.base_model import BaseModel


class ZonosTTSModel(BaseModel):
    def load_model(self, model_path: str, device: str):
        model = Zonos.from_pretrained(model_path, device=device)

        self.device = device
        self.model = model

    def inference(self, texts: []):
        wav, sampling_rate = torchaudio.load("assets/exampleaudio.mp3")
        speaker = self.model.make_speaker_embedding(wav, sampling_rate)

        cond_dict = make_cond_dict(text="Hello, world!", speaker=speaker, language="en-us")
        conditioning = self.model.prepare_conditioning(cond_dict)

        codes = self.model.generate(conditioning)

        wavs = self.model.autoencoder.decode(codes).cpu()
        torchaudio.save("sample.wav", wavs[0], self.model.autoencoder.sampling_rate)

    def release(self):
        del self.model
