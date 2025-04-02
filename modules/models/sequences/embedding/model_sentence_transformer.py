import torch
from typing import List, Union
from numpy import ndarray
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from modules.models.sequences.embedding.base_model import BaseModel


class SentenceTransformerModel(BaseModel):
    def load_model(self, model_path: str, device: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(model_path, device=device, trust_remote_code=True)

        self.device = device
        self.model = model

    def encode(self,
               sentences: Union[str, List[str]],
               model_name: str = "m3e-large",
               return_numpy: bool = True):

        if isinstance(sentences, str):
            sentences = [sentences]

        if model_name == "Yinka":
            embeddings = self.model.encode(sentences, normalize_embeddings=False)
            n_dims = 768
            embeddings = normalize(embeddings[:, :n_dims])
        elif model_name == "jina-embeddings-v2-base-zh":
            self.model.max_seq_length = 768
            embeddings = self.model.encode(sentences, normalize_embeddings=True)
        else:
            embeddings = self.model.encode(sentences, normalize_embeddings=True)

        if return_numpy and not isinstance(embeddings, ndarray):
            embeddings = embeddings.numpy()

        return embeddings

    def release(self):
        del self.model
