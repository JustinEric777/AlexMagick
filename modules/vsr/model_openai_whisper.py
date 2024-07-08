import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from modules.vsr.base_model import BaseModel


class OpenAIWhisperModel(BaseModel):
    def load_model(self, model_path: str):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        processor = AutoProcessor.from_pretrained(AutoModelForSpeechSeq2Seq)

        self.device = device
        self.dtype = dtype
        self.model = model
        self.processor = processor

    def generate(self, audio: str):
        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            truncation=False,
            padding="longest",
            return_attention_mask=True,
        ).to(self.device, dtype=self.dtype)

        generate_kwargs = {
            "max_new_tokens": 448,
            "num_beams": 1,
            "condition_on_prev_tokens": False,
            "compression_ratio_threshold": 1.35,
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "return_timestamps": True,
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

        return predict_text


