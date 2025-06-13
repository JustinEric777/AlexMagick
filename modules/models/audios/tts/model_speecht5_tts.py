import os.path
import time
import torch
import torchaudio
import soundfile as sf
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from modules.models.audios.tts.base_model import BaseModel, AUDIO_PATH


class SpeechT5TTSModel(BaseModel):
    def __init__(self):
        self.speaker_embedding = None

    def load_model(self, model_path: str, device: str):
        model = SpeechT5ForTextToSpeech.from_pretrained(model_path)
        processor = SpeechT5Processor.from_pretrained(model_path)

        speaker_embedding_path = os.path.join(model_path, "spkrec-xvect-voxceleb")
        speaker_embedding = EncoderClassifier.from_hparams(
            source=speaker_embedding_path,
            savedir=speaker_embedding_path
        )

        vocoder_model_path = os.path.join(model_path, "speecht5_hifigan")
        vocoder = SpeechT5HifiGan.from_pretrained(vocoder_model_path)

        self.model = model
        self.processor = processor
        self.vocoder = vocoder
        self.speaker_embedding = speaker_embedding

    def inference(self, texts: [], sample_wav: str = None):
        if sample_wav is None:
            sample_wav = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cosyvoice/examples/zero_shot_prompt.wav")
        else:
            sample_wav = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cosyvoice/examples/zero_shot_prompt.wav")

        wav, sampling_rate = torchaudio.load(sample_wav)
        with torch.no_grad():
            embeddings = self.speaker_embedding.encode_batch(wav)
            embeddings = F.normalize(embeddings, dim=2)
            speaker_embeddings = embeddings.squeeze().unsqueeze(0)

        input_ids = self.processor(
            text=texts,
            return_tensors="pt"
        )

        wavs = self.model.generate_speech(
            input_ids["input_ids"],
            speaker_embeddings,
            vocoder=self.vocoder
        )

        audio_path = os.path.join(AUDIO_PATH, f"speecht5_tts_{int(time.time() * 1000)}.wav")
        sf.write(audio_path, wavs.numpy(), samplerate=16000)

        return audio_path

    def release(self):
        del self.model
        del self.vocoder
        del self.processor
        del self.speaker_embedding
