from modules.models.audios.asr.base_model import BaseModel


class MetaWev2VecConformer(BaseModel):
    def load_model(self, model_path: str, device: str):
        pass

    def generate(self, audio: str):
        pass