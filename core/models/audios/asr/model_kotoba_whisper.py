import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from core.models.audios.asr.base_model import BaseModel

class KotobaWhisperModel(BaseModel):
    def load_model(self, model_path: str, device: str, **kwargs):
        if not device or device == "AUTO":
             device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif device == "cuda":
             device = "cuda:0"
             
        dtype = torch.float16 if torch.cuda.is_available() and device != "cpu" else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path, 
            dtype=dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_path)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            dtype=dtype,
            device=device,
        )
        self.device = device

    def _generate(self, audio_path: str):
        result = self.pipe(audio_path, generate_kwargs={"language": "japanese"})
        return result["text"]

    def release(self):
        del self.pipe
