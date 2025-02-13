import torch
from numpy import ndarray
from typing import List, Union
from transformers import AutoModel, AutoTokenizer
from modules.models.sequences.embedding.base_model import BaseModel


class TransformerSentenceModel(BaseModel):
    def load_model(self, model_path: str, device: str):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)

        model.eval()
        model.to(device)

        self.pooling_method = 'cls'
        self.device = device
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def encode(self,
               sentences: Union[str, List[str]],
               model_name: str = "",
               batch_size: int = 32,
               max_length: int = 512,
               normalize_to_unit: bool = True,
               return_numpy: bool = True,
               enable_tqdm: bool = True,
               query_instruction: str = ""):

        if isinstance(sentences, str):
            sentences = [sentences]

        embeddings_collection = []
        for sentence_id in range(0, len(sentences), batch_size):
            if isinstance(query_instruction, str) and len(query_instruction) > 0:
                sentence_batch = [query_instruction + sent for sent in
                                  sentences[sentence_id:sentence_id + batch_size]]
            else:
                sentence_batch = sentences[sentence_id:sentence_id + batch_size]
            inputs = self.tokenizer(
                sentence_batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(self.device)
            inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs_on_device, return_dict=True)

            if self.pooling_method == "cls":
                embeddings = outputs.last_hidden_state[:, 0]
            elif self.pooling_method == "mean":
                attention_mask = inputs_on_device['attention_mask']
                last_hidden = outputs.last_hidden_state
                embeddings = (last_hidden * attention_mask.unsqueeze(-1).float()).sum(1) / attention_mask.sum(
                    -1).unsqueeze(-1)
            else:
                raise NotImplementedError

            if normalize_to_unit:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            embeddings_collection.append(embeddings.cpu())

        embeddings = torch.cat(embeddings_collection, dim=0)

        if return_numpy and not isinstance(embeddings, ndarray):
            embeddings = embeddings.numpy()

        return embeddings
