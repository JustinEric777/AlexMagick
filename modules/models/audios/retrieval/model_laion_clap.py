from typing import List, Union
import torch
import torchaudio
import torchaudio.functional as F
from transformers import ClapModel, ClapProcessor
from modules.models.audios.retrieval.base_model import BaseModel


class LaionClapModel(BaseModel):
    def load_model(self, model_path: str, device: str):
        model = ClapModel.from_pretrained(model_path).to(device.lower())
        processor = ClapProcessor.from_pretrained(model_path)

        self.model = model
        self.processor = processor
        self.device = device

    @torch.no_grad()
    def get_text_features(self, texts: Union[str, List[str]]):
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True
        )

        embeddings = self.model.get_text_features(**inputs)
        return embeddings

    @torch.no_grad()
    def get_audio_features(self, audios: Union[str, List[str]]):
        waveform, sample_rate = torchaudio.load(audios)
        if waveform.shape[0] > 0:
             waveform = waveform.mean(dim=0)

        target_sample_rate = 48000
        if sample_rate != target_sample_rate:
            waveform = F.resample(waveform, orig_freq=sample_rate, new_freq=target_sample_rate)

        inputs = self.processor(
            audios=waveform,
            max_length_s=10,
            return_tensors="pt",
            padding=True
        )

        embeddings = self.model.get_audio_features(**inputs)
        return embeddings

    @torch.no_grad()
    def calculate_sim(self, texts: Union[str, List[str]], audios: Union[str, List[str]]):
        waveform, sample_rate = torchaudio.load(audios)
        if waveform.shape[0] > 0:
            waveform = waveform.mean(dim=0)

        target_sample_rate = 48000
        if sample_rate != target_sample_rate:
            waveform = F.resample(waveform, orig_freq=sample_rate, new_freq=target_sample_rate)

        inputs = self.processor(
            text=texts,
            audios=waveform,
            return_tensors="pt",
            padding=True
        )

        outputs = self.model(**inputs)
        logits_per_audio = outputs.logits_per_audio
        probs = logits_per_audio.softmax(dim=-1)

        return probs.tolist()

    def release(self):
        del self.model
        del self.processor
        del self.device



