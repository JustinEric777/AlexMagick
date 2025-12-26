from typing import List, Union
import torch
from msclap import CLAP
from core.models.audios.retrieval.base_model import BaseModel


class AnySpeechClapIpaModel(BaseModel):
    def load_model(self, model_path: str, device: str):
        model = CLAP(model_fp=model_path, version='2023', use_cuda=False)
        self.model = model
        self.device = device.lower()

    @torch.no_grad()
    def get_text_features(self, texts: Union[str, List[str]]):
        return self.model.get_text_embeddings(class_labels=texts)

    @torch.no_grad()
    def get_audio_features(self, audios: Union[str, List[str]]):
        return self.model.get_audio_embeddings(audio_files=audios)

    @torch.no_grad()
    def calculate_sim(self, texts: Union[str, List[str]], audios: Union[str, List[str]]):
        text_embeddings = self.model.get_text_embeddings(class_labels=texts)
        audio_embeddings = self.model.get_audio_embeddings(audio_files=[audios])
        probs = self.model.compute_similarity(audio_embeddings, text_embeddings)
        return probs.softmax(dim=-1).tolist()

    def release(self):
        del self.model

