import os.path
import re
import time
import torchaudio
import soundfile as sf
from modules.models.audios.tts.base_model import BaseModel, AUDIO_PATH
from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import (
    load_model,
    infer_process,
    load_vocoder
)


def load_f5tts(model_path: str):
    ckpt_path = os.path.join(model_path, "F5TTS_v1_Base/model_1250000.safetensors")
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)

    return load_model(DiT, model_cfg, ckpt_path)


def load_e2tts(model_path: str):
    ckpt_path = os.path.join(model_path, "E2TTS_v1_Base/model_1200000.safetensors")
    model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4, text_mask_padding=False, pe_attn_head=1)

    return load_model(UNetT, model_cfg, ckpt_path)


class F5TTSModel(BaseModel):
    def load_model(self, model_path: str, device: str, vocoder_name="vocos"):
        vocoder_model_path = os.path.join(model_path, "vocos-mel-24khz")
        vocoder = load_vocoder(
            vocoder_name=vocoder_name,
            is_local=True,
            local_path=vocoder_model_path,
            device=device.lower()
        )

        model = load_f5tts(model_path)

        self.device = device.lower()
        self.model = model
        self.vocoder = vocoder

    def inference(self, text: str, sample_wav: str = None):
        if sample_wav is None:
            sample_wav = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f5_tts/infer/examples/basic/basic_ref_zh.wav")
        else:
            sample_wav = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f5_tts/infer/examples/basic/basic_ref_zh.wav")

        gen_text = re.sub(r"\[(\w+)]", "", text).strip()
        final_wave, final_sample_rate, combined_spectrogram = infer_process(
            sample_wav,
            text,
            gen_text,
            self.model,
            self.vocoder,
            cross_fade_duration=0.15,
            nfe_step=32,
            speed=1.0,
            cfg_strength=2.0,
            sway_sampling_coef=-1.0,
        )

        audio_path = os.path.join(AUDIO_PATH, f"f5_tts_{int(time.time() * 1000)}.wav")
        sf.write(audio_path, final_wave, final_sample_rate)

        return audio_path

    def release(self):
        del self.model
        del self.vocoder
