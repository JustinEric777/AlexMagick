import torch
from typing import List, Union
from numpy import ndarray
from sentence_transformers import SentenceTransformer
from modules.models.sequences.embedding.base_model import BaseModel


class SentenceTransformerModel(BaseModel):
    def load_model(self, model_path: str):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(model_path, device=device)

        self.device = device
        self.model = model

    def encode(self,
               sentences: Union[str, List[str]],
               return_numpy: bool = True):

        if isinstance(sentences, str):
            sentences = [sentences]

        embeddings = self.model.encode(sentences)

        if return_numpy and not isinstance(embeddings, ndarray):
            embeddings = embeddings.numpy()

        return embeddings
