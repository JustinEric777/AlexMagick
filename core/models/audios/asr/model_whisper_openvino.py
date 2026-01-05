import torch
from transformers import AutoTokenizer, AutoProcessor, pipeline
from core.models.audios.asr.base_model import BaseModel
import logging

logger = logging.getLogger(__name__)

class WhisperOpenVINOModel(BaseModel):
    def load_model(self, model_path: str, device: str, **kwargs):
        try:
            from optimum.intel import OVModelForSpeechSeq2Seq
        except ImportError:
            logger.error("optimum-intel is not installed. Please install it to use OpenVINO models.")
            raise

        # Map device names to OpenVINO compatible ones
        if not device or device == "AUTO":
             device = "CPU" # Default to CPU for OpenVINO if auto
        
        # OVModel usually handles device via .to() or export=False + load logic
        # For pre-exported models (int4-ov), we load directly.
        
        logger.info(f"Loading OpenVINO model from {model_path} on {device}")
        
        self.model = OVModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            device=device
        )
        
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=30,
            batch_size=16,
        )
        self.device = device

    def _generate(self, audio: str):
        result = self.pipe(audio)
        return result["text"]

    def release(self):
        del self.pipe
        del self.model
        del self.processor
