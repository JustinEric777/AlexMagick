import os
import time
from modules.models.audios.tts.base_model import BaseModel, AUDIO_PATH
from dia.model import Dia


class DiaTTSModel(BaseModel):
    def load_model(self, model_path: str, device: str):
        config_path = os.path.join(model_path, "config.json")
        checkpoint_path = os.path.join(model_path, "dia-v0_1.pth")
        model = Dia.from_local(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device.lower(),
            load_dac=False
        )

        self.device = device
        self.model = model

    def inference(self, text: str, sample_wav: str = None):
        if sample_wav is None:
            sample_wav = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cosyvoice/examples/zero_shot_prompt.wav")
        else:
            sample_wav = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cosyvoice/examples/zero_shot_prompt.wav")

        output = self.model.generate(
            text, use_torch_compile=False, verbose=True
        )

        audio_path = os.path.join(AUDIO_PATH, f"dia_tts_{int(time.time() * 1000)}.wav")
        self.model.save_audio(audio_path, output)

        return audio_path

    def release(self):
        del self.device
        del self.model
