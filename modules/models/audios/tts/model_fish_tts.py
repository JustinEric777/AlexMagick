import os.path
import torch
import time
import torchaudio
import numpy as np
import soundfile as sf
from fish_speech.models.vqgan.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import load_model as load_semantic_model
from fish_speech.models.text2semantic.inference import generate_long as semantic_generate
from modules.models.audios.tts.base_model import BaseModel, AUDIO_PATH


class FishTTSModel(BaseModel):
    def load_model(self, model_path: str, device: str):
        decoder_model_path = os.path.join(model_path, "firefly-gan-vq-fsq-8x1024-21hz-generator.pth")
        vocoder = load_decoder_model(
            config_name="firefly_gan_vq",
            checkpoint_path=decoder_model_path,
            device=device.lower(),
        )

        model, tokenizer = load_semantic_model(
            model_path,
            device=device.lower(),
            precision=torch.float32,
            compile=False
        )

        self.model = model
        self.tokenizer = tokenizer
        self.vocoder = vocoder
        self.device = device.lower()

    def inference(self, texts: [], sample_wav: str = None, seed: int = 42):
        if sample_wav is None:
            sample_wav = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cosyvoice/examples/zero_shot_prompt.wav")
        else:
            sample_wav = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cosyvoice/examples/zero_shot_prompt.wav")

        audio, sample_rate = torchaudio.load(sample_wav)
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        audios = torchaudio.functional.resample(
            audio, sample_rate, self.vocoder.spec_transform.sample_rate
        )[None].to(self.device)

        audio_lengths = torch.tensor([audios.shape[2]], device=self.device, dtype=torch.long)
        speaker_embedding = self.vocoder.encode(audios, audio_lengths)[0][0]

        torch.manual_seed(seed)
        generator = semantic_generate(
            model=self.model,
            device=self.device,
            decode_one_token=self.tokenizer,
            text=texts,
            prompt_text="The text corresponding to reference audio",
            prompt_tokens=speaker_embedding,
        )

        output_dir = ""
        idx = 0
        codes = []
        for response in generator:
            print("response = ", response)
            if response.action == "sample":
                codes.append(response.codes)
                print(f"Sampled text: {response.text}")
            elif response.action == "next":
                if codes:
                    codes_npy_path = os.path.join(output_dir, f"codes_{idx}.npy")
                    np.save(codes_npy_path, torch.cat(codes, dim=1).cpu().numpy())
                    print(f"Saved codes to {codes_npy_path}")
                codes = []
                idx += 1
            else:
                print(f"Error: {response}")

        indices = np.load(output_dir)
        indices = torch.from_numpy(indices).to(self.device).long()
        feature_lengths = torch.tensor([indices.shape[1]], device=self.device)
        fake_audios, _ = self.vocoder.decode(
            indices=indices[None], feature_lengths=feature_lengths
        )

        # Save audio
        audio_path = os.path.join(AUDIO_PATH, f"fish_speech_{int(time.time() * 1000)}.wav")
        fake_audio = fake_audios[0, 0].float().cpu().numpy()
        sf.write(audio_path, fake_audio, self.vocoder.spec_transform.sample_rate)

        return audio_path

    def release(self):
        del self.vocoder
        del self.model
