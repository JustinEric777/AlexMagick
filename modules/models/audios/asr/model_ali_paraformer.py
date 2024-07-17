from modules.models.audios.asr.base_model import BaseModel


class AliParaformerModel(BaseModel):
    def load_model(self, model_path: str):
        pass

    def generate(self, audio: str):
        pass