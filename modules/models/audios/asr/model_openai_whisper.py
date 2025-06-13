import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from modules.models.audios.asr.base_model import BaseModel


class OpenAIWhisperModel(BaseModel):
    def load_model(self, model_path: str, device: str):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        processor = AutoProcessor.from_pretrained(model_path)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=30,
            torch_dtype=dtype,
            device=device,
        )

        self.device = device
        self.dtype = dtype
        self.model = model
        self.pipeline = pipe
        self.processor = processor

    def generate(self, audio: str):
        audio, sampling_rate = torchaudio.load(audio)
        result = self.pipeline(audio[0])

        return result["text"]

    def release(self):
        del self.pipeline
        del self.model
        del self.processor


