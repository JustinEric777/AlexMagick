import torch
import logging
import torchaudio
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, WhisperFeatureExtractor
import transformers.models.whisper.modeling_whisper as whisper_module
from core.models.audios.asr.base_model import BaseModel

# Monkeypatch for GLM-ASR compatibility with newer transformers
if not hasattr(whisper_module, "WhisperFlashAttention2"):
    whisper_module.WhisperFlashAttention2 = whisper_module.WhisperAttention

logger = logging.getLogger(__name__)

WHISPER_FEAT_CFG = {
    "chunk_length": 30,
    "feature_extractor_type": "WhisperFeatureExtractor",
    "feature_size": 128,
    "hop_length": 160,
    "n_fft": 400,
    "n_samples": 480000,
    "nb_max_frames": 3000,
    "padding_side": "right",
    "padding_value": 0.0,
    "processor_class": "WhisperProcessor",
    "return_attention_mask": False,
    "sampling_rate": 16000,
}

class GLMASRModel(BaseModel):
    def load_model(self, model_path: str, device: str, **kwargs):
        if not device or device == "AUTO":
             device = "cuda:0" if torch.cuda.is_available() else "cpu"
             
        logger.info(f"Loading GLM-ASR model from {model_path} on {device}")
        
        self.feature_extractor = WhisperFeatureExtractor(**WHISPER_FEAT_CFG)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # GLM-ASR requires AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            device_map=device,
            dtype=torch.bfloat16 if device != "cpu" and torch.cuda.is_available() else torch.float32
        )
        
        # Patch prepare_inputs_for_generation to fix AttributeError
        self._patch_glmasr_model(self.model)
        
        self.model.eval()
        self.device = device
        self.merge_factor = self.model.config.merge_factor if hasattr(self.model.config, "merge_factor") else 4

    def _patch_glmasr_model(self, model):
        def fixed_prepare_inputs_for_generation(
            self,
            *args,
            past_key_values=None,
            attention_mask=None,
            position_ids=None,
            use_cache=None,
            is_first_forward=True,
            **kwargs,
        ):
            from transformers import LlamaForCausalLM
            prepared = LlamaForCausalLM.prepare_inputs_for_generation(
                self,
                *args,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=use_cache,
                is_first_forward=is_first_forward,
                **kwargs,
            )
            for key, value in kwargs.items():
                if key not in prepared and key.startswith("audio"):
                    prepared[key] = value
            
            # FIX: Add check for past_key_values[0][0] is not None
            if is_first_forward and past_key_values is not None and len(past_key_values) > 0:
                if past_key_values[0][0] is not None:
                    cached_len = past_key_values[0][0].shape[2]
                    prepared["input_ids"] = prepared["input_ids"][:, cached_len:]
                    if "position_ids" in prepared:
                        prepared["position_ids"] = prepared["position_ids"][:, cached_len:]
            
            if not is_first_forward:
                prepared["audios"] = None
            return prepared

        # Bind the method to the class
        logger.info(f"Original prepare_inputs_for_generation: {model.__class__.prepare_inputs_for_generation}")
        model.__class__.prepare_inputs_for_generation = fixed_prepare_inputs_for_generation
        logger.info(f"Patched prepare_inputs_for_generation: {model.__class__.prepare_inputs_for_generation}")
        logger.info("Patched GLM-ASR prepare_inputs_for_generation")

    def get_audio_token_length(self, seconds, merge_factor=2):
        def get_T_after_cnn(L_in, dilation=1):
            for padding, kernel_size, stride in [(1,3,1), (1,3,2)]:
                L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
                L_out = 1 + L_out // stride
                L_in = L_out
            return L_out

        mel_len = int(seconds * 100)
        audio_len_after_cnn = get_T_after_cnn(mel_len)
        audio_token_num = (audio_len_after_cnn - merge_factor) // merge_factor + 1
        audio_token_num = min(audio_token_num, 1500 // merge_factor)
        return audio_token_num

    def build_prompt(self, audio_path: str, chunk_seconds: int = 30) -> dict:
        wav, sr = torchaudio.load(audio_path)
        wav = wav[:1, :]
        if sr != self.feature_extractor.sampling_rate:
            wav = torchaudio.transforms.Resample(sr, self.feature_extractor.sampling_rate)(wav)

        tokens = []
        tokens += self.tokenizer.encode("<|user|>")
        tokens += self.tokenizer.encode("\n")

        audios = []
        audio_offsets = []
        audio_length = []
        chunk_size = chunk_seconds * self.feature_extractor.sampling_rate
        
        # Handle empty audio
        if wav.shape[1] == 0:
            raise ValueError("Audio is empty")

        for start in range(0, wav.shape[1], chunk_size):
            chunk = wav[:, start : start + chunk_size]
            mel = self.feature_extractor(
                chunk.numpy(),
                sampling_rate=self.feature_extractor.sampling_rate,
                return_tensors="pt",
                padding="max_length",
            )["input_features"]
            audios.append(mel)
            seconds = chunk.shape[1] / self.feature_extractor.sampling_rate
            num_tokens = self.get_audio_token_length(seconds, self.merge_factor)
            tokens += self.tokenizer.encode("<|begin_of_audio|>")
            audio_offsets.append(len(tokens))
            tokens += [0] * num_tokens
            tokens += self.tokenizer.encode("<|end_of_audio|>")
            audio_length.append(num_tokens)

        if not audios:
             raise ValueError("Audio processing failed.")

        tokens += self.tokenizer.encode("<|user|>")
        tokens += self.tokenizer.encode("\nPlease transcribe this audio into text")
        tokens += self.tokenizer.encode("<|assistant|>")
        tokens += self.tokenizer.encode("\n")

        batch = {
            "input_ids": torch.tensor([tokens], dtype=torch.long),
            "audios": torch.cat(audios, dim=0),
            "audio_offsets": [audio_offsets],
            "audio_length": [audio_length],
            "attention_mask": torch.ones(1, len(tokens), dtype=torch.long),
        }
        return batch

    def _generate(self, audio_path: str):
        batch = self.build_prompt(audio_path)
        
        device = torch.device(self.device)
        tokens = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        audios = batch["audios"].to(device)
        
        model_inputs = {
            "inputs": tokens,
            "attention_mask": attention_mask,
            "audios": audios.to(self.model.dtype),
            "audio_offsets": batch["audio_offsets"],
            "audio_length": batch["audio_length"],
        }
        
        with torch.no_grad():
            generated = self.model.generate(
                **model_inputs,
                max_new_tokens=256,
                do_sample=False,
            )
            
        prompt_len = tokens.size(1)
        transcript_ids = generated[0, prompt_len:].cpu().tolist()
        transcript = self.tokenizer.decode(transcript_ids, skip_special_tokens=True).strip()
        
        return transcript

    def release(self):
        del self.model
        del self.tokenizer
        del self.feature_extractor
