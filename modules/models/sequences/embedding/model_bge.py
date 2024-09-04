import torch
from tqdm import tqdm
import numpy as np
from typing import List, Union, cast
from base_model import BaseModel
from transformers import AutoModel, AutoTokenizer


class BgeModel(BaseModel):
    def load_model(self, model_path: str):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path)

        model.eval()
        model.to(device)

        self.pooling_method = 'cls'
        self.device = device
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def encode(self,
               sentences: Union[str, List[str]],
               batch_size: int = 256,
               max_length: int = 512,
               return_numpy: bool = True):

        if isinstance(sentences, str):
            sentences = [sentences]

        with torch.no_grad():
            all_embeddings = []
            for start_index in tqdm(range(0, len(sentences), batch_size), desc="Inference Embeddings",
                                    disable=len(sentences) < 256):
                sentences_batch = sentences[start_index:start_index + batch_size]
                inputs = self.tokenizer(
                    sentences_batch,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=max_length,
                ).to(self.device)
                last_hidden_state = self.model(**inputs, return_dict=True).last_hidden_state
                embeddings = self.pooling(last_hidden_state, inputs['attention_mask'])
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
                embeddings = cast(torch.Tensor, embeddings)

                if return_numpy:
                    embeddings = embeddings.cpu().numpy()
                all_embeddings.append(embeddings)

            if return_numpy:
                all_embeddings = np.concatenate(all_embeddings, axis=0)
            else:
                all_embeddings = torch.cat(all_embeddings, dim=0)

            return all_embeddings

    def pooling(self,
                last_hidden_state: torch.Tensor,
                attention_mask: torch.Tensor = None):
        if self.pooling_method == 'cls':
            return last_hidden_state[:, 0]
        elif self.pooling_method == 'mean':
            s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d