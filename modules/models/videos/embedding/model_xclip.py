import torch
import av
from typing import Union, List
import numpy as np
from transformers import AutoModel, AutoProcessor
from modules.models.videos.embedding.base_model import BaseModel


def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


class XClipModel(BaseModel):
    def load_model(self, model_path: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)

        model.eval()
        model.to(device)

        self.model = model
        self.tokenizer = processor
        self.device = device

    @torch.no_grad()
    def text_encode(self, sentences: Union[str, List[str]]):
        if isinstance(sentences, str):
            sentences = [sentences]

        inputs = self.tokenizer(
            text=sentences,
            return_tensors="pt",
            padding=True
        )

        embeddings = self.model.get_text_features(**inputs)
        embeddings = embeddings.numpy()

        return embeddings

    @torch.no_grad()
    def video_encode(self, video_path: str):
        container = av.open(video_path)
        indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
        video = read_video_pyav(container, indices)

        inputs = self.tokenizer(videos=list(video), return_tensors="pt")

        embeddings = self.model.get_video_features(**inputs)
        embeddings = embeddings.numpy()

        return embeddings

