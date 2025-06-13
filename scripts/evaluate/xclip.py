import os.path

import av
import torch
import numpy as np

from transformers import AutoProcessor, AutoModel

np.random.seed(0)

data_path = "/aigc/data/wangjing/datasets/msvd_data_test_1000.csv"
root_path = "/aigc/data/wangjing/datasets/YouTubeClips/"
model_path = "/aigc/data/wangjing/models/xclip-base-patch32"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
model.eval()
model.to(device)


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


def get_videos():
    videos, labels = [], {}

    with open(data_path, "r") as file:
        contents = file.read()
        lines_arr = contents.split("\n")
        lines_arr = lines_arr[999:]
        for line in lines_arr:
            line_arr = line.split(",")
            if len(line_arr) > 1:
                file_name = line_arr[0]
                text_label = line_arr[1]
                file_path = os.path.join(root_path, file_name+".avi")
                if os.path.exists(file_path):
                    container = av.open(file_path)
                    indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
                    video = read_video_pyav(container, indices)

                    labels[file_name] = text_label
                    videos.append(video)

    return videos, labels


videos, labels = get_videos()
texts = ["a woman is water skiing"]

print(videos[1])


inputs = processor(
    text=texts,
    videos=list(videos[0]),
    return_tensors="pt",
    padding=True
)

# forward pass
with torch.no_grad():
    outputs = model(**inputs)


