import os.path
from typing import List, Union
import torch
import torchaudio
import torch.nn.functional as F
import torchaudio.functional as AF
from transformers import DebertaV2Tokenizer, AutoProcessor
from clap.encoders import SpeechEncoder, PhoneEncoder
from modules.models.audios.retrieval.base_model import BaseModel


class ClapIpaModel:
    speech_encoder: None
    phone_encoder: None

    def __init__(self, speech_encoder, phone_encoder):
        self.speech_encoder = speech_encoder
        self.phone_encoder = phone_encoder


class ClapIpaProcessor:
    text_processor: None
    audio_processor: None

    def __init__(self, audio_processor, text_processor):
        self.audio_processor = audio_processor
        self.text_processor = text_processor


class AnySpeechClapIpaModel(BaseModel):
    def load_model(self, model_path: str, device: str):
        speech_encoder_path = os.path.join(model_path, f"{os.path.basename(model_path)}-speech")
        speech_encoder = SpeechEncoder.from_pretrained(speech_encoder_path).eval().to(device.lower())

        phone_encoder_path = os.path.join(model_path, f"{os.path.basename(model_path)}-phone")
        phone_encoder = PhoneEncoder.from_pretrained(phone_encoder_path).eval().to(device.lower())

        text_tokenizer_path = os.path.join(model_path, "IPATokenizer")
        tokenizer = DebertaV2Tokenizer.from_pretrained(text_tokenizer_path)

        speech_tokenizer_path = os.path.join(model_path, "whisper")
        processor = AutoProcessor.from_pretrained(speech_tokenizer_path)

        self.model = ClapIpaModel(speech_encoder, phone_encoder)
        self.processor = ClapIpaProcessor(processor, tokenizer)
        self.device = device

    @torch.no_grad()
    def get_text_features(self, texts: Union[str, List[str]]):
        inputs = self.processor.text_processor(
            texts,
            return_tensors="pt",
            padding=True
        )

        embeddings = self.model.phone_encoder(**inputs)
        return embeddings

    @torch.no_grad()
    def get_audio_features(self, audios: Union[str, List[str]]):
        waveform, sample_rate = torchaudio.load(audios)
        if waveform.shape[0] > 0:
            waveform = waveform.mean(dim=0)

        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            waveform = AF.resample(waveform, orig_freq=sample_rate, new_freq=target_sample_rate)
            sample_rate = target_sample_rate

        inputs = self.processor.audio_processor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
            truncation=True,
            padding="longest",
            return_attention_mask=True,
        )

        embeddings = self.model.speech_encoder(**inputs.input_features)
        return embeddings

    @torch.no_grad()
    def calculate_sim(self, texts: Union[str, List[str]], audios: Union[str, List[str]]):
        inputs_text = self.processor.text_processor(
            texts,
            return_tensors="pt",
            padding=True,
            return_attention_mask=False,
            return_length=True,
            return_token_type_ids=False,
            add_special_tokens=False
        )
        inputs_text = torch.tensor(inputs_text["input_ids"])

        waveform, sample_rate = torchaudio.load(audios)
        if waveform.shape[0] > 0:
            waveform = waveform.mean(dim=0)
        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            waveform = AF.resample(waveform, orig_freq=sample_rate, new_freq=target_sample_rate)
            sample_rate = target_sample_rate

        inputs_audio = self.processor.audio_processor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
            chunk_length_s=30,
            truncation=True,
            padding=True,
            return_attention_mask=True
        )

        speech_embeddings = self.model.speech_encoder(**inputs_audio, return_dict=True).last_hidden_state
        text_embeddings = self.model.phone_encoder(inputs_text).last_hidden_state.squeeze(0)

        speech_embeddings = speech_embeddings.mean(dim=1)
        text_embeddings = text_embeddings.mean(dim=1)
        probs = F.cosine_similarity(speech_embeddings, text_embeddings, dim=-1)

        return [probs.tolist()]

    def release(self):
        del self.model
        del self.processor
        del self.device



