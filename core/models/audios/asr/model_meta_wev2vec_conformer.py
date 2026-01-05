import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from core.models.audios.asr.base_model import BaseModel

class MetaWev2VecConformer(BaseModel):
    def load_model(self, model_path: str, device: str, **kwargs):
        if not device or device == "AUTO":
             device = "cuda:0" if torch.cuda.is_available() else "cpu"
             
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path)
        self.model.to(device)
        self.device = device

    def _generate(self, audio: str):
        # Load audio
        speech, sample_rate = torchaudio.load(audio)
        
        # Resample if necessary (Wav2Vec2 usually expects 16kHz)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            speech = resampler(speech)
            sample_rate = 16000
            
        # Ensure single channel
        if speech.shape[0] > 1:
            speech = speech.mean(dim=0, keepdim=True)
            
        input_values = self.processor(speech[0], return_tensors="pt", sampling_rate=16000).input_values
        input_values = input_values.to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_values).logits
            
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return transcription

    def release(self):
        del self.model
        del self.processor
