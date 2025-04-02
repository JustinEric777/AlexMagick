import torchaudio
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.cli.cosyvoice import CosyVoice
from modules.models.audios.tts.base_model import BaseModel


class CosyVoiceModel(BaseModel):
    def load_model(self, model_path: str, device: str):
        model = CosyVoice(model_path)

        self.device = device
        self.model = model

    def inference(self, texts: []):
        prompt_speech_16k = load_wav('zero_shot_prompt.wav', 16000)
        output = self.model.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k)
        torchaudio.save('zero_shot.wav', output['tts_speech'], 22050)

    def release(self):
        del self.model