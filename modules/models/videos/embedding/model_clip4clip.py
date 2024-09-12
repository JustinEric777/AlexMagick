import torch
import cv2
import numpy as np
from PIL import Image
from typing import Union, List
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from transformers import CLIPTokenizer, CLIPTextModelWithProjection
from modules.models.videos.embedding.base_model import BaseModel


def video2image(video_path, frame_rate=1.0, size=224):
    def preprocess(size, n_px):
        return Compose([
            Resize(size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(size),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])(n_px)

    cap = cv2.VideoCapture(video_path)
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps < 1:
        images = np.zeros([3, size, size], dtype=np.float32)
        print("ERROR: problem reading video file: ", video_path)
    else:
        total_duration = (frameCount + fps - 1) // fps
        start_sec, end_sec = 0, total_duration
        interval = fps / frame_rate
        frames_idx = np.floor(np.arange(start_sec*fps, end_sec*fps, interval))
        ret = True
        images = np.zeros([len(frames_idx), 3, size, size], dtype=np.float32)

        for i, idx in enumerate(frames_idx):
            cap.set(cv2.CAP_PROP_POS_FRAMES , idx)
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last_frame = i
            images[i,:,:,:] = preprocess(size, Image.fromarray(frame).convert("RGB"))

        images = images[:last_frame+1]
    cap.release()
    video_frames = torch.tensor(images)

    return video_frames


class Clip4ClipModel(BaseModel):
    def load_model(self, model_path: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CLIPTextModelWithProjection.from_pretrained(model_path)
        tokenizer = CLIPTokenizer.from_pretrained(model_path)

        model.eval()
        model.to(device)

        self.device = device
        self.model = model
        self.tokenizer = tokenizer

    def text_encode(self, sentences: Union[str, List[str]]):
        if isinstance(sentences, str):
            sentences = [sentences]

        inputs = self.tokenizer(text=sentences, return_tensors="pt")

        embedding = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        embedding = torch.nn.functional.normalize(embedding, dim=-1)
        embedding = embedding.numpy()

        return embedding

    def video_encode(self, video_path: str):
        video = video2image(video_path)
        output = self.model(video)
        embedding = output["image_embeds"]

        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        embedding = torch.mean(embedding, dim=0)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        # embedding = torch.nn.functional.normalize(embedding, dim=-1)
        embedding = embedding.numpy()

        return embedding

