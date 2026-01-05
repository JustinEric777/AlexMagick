from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from core.models.audios.asr.base_model import BaseModel


class AliFunASRModel(BaseModel):
    def load_model(self, model_path: str, device: str, **kwargs):
        # Allow disabling VAD or specifying it via kwargs, default to None to avoid auto-download
        vad_model = kwargs.get("vad_model", None)
        vad_kwargs = kwargs.get("vad_kwargs", {})
        
        # If device passed is "cuda:0", map to "cuda" for AutoModel if needed, 
        # but AutoModel usually handles "cuda:0" fine.
        
        # Check for remote code (model.py) in the model directory
        remote_code = None
        trust_remote_code = False
        
        import os
        potential_code = os.path.join(model_path, "model.py")
        if os.path.exists(potential_code):
            remote_code = potential_code
            trust_remote_code = True

        model = AutoModel(
            model=model_path,
            vad_model=vad_model,
            vad_kwargs=vad_kwargs,
            device=device,
            trust_remote_code=trust_remote_code,
            remote_code=remote_code,
        )

        self.model = model
        self.device = device

    def _generate(self, audio: str):
        res = self.model.generate(
            input=f"{audio}",
            cache={},
            language="auto",
            use_itn=True,
            batch_size_s=4,
            merge_vad=True,
            merge_length_s=15,
        )

        text = rich_transcription_postprocess(res[0]["text"])
        return text
