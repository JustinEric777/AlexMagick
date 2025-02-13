import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
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

        self.device = device
        self.dtype = dtype
        self.model = model
        self.processor = processor

    def generate(self, audio: str):
        audio_info, sampling_rate = torchaudio.load(audio)
        inputs = self.processor(
            audio_info[0],
            sampling_rate=sampling_rate,
            return_tensors="pt",
            truncation=False,
            padding="longest",
            return_attention_mask=True,
        ).to(self.device, dtype=self.dtype)

        generate_kwargs = {
            "max_new_tokens": 448,
            "num_beams": 1,
            "return_timestamps": True,
            "do_sample": True
        }

        predict_ids = self.model.generate(
            **inputs,
            **generate_kwargs
        )
        predict_text = self.processor.batch_decode(
            predict_ids,
            skip_special_tokens=True,
            decode_with_timestamps=False
        )

        return predict_text[0]


