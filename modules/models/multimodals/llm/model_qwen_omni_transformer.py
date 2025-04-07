import os.path
import time
import ffmpeg
import soundfile as sf
from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
from .qwen_omni_utils import process_mm_info
from gradio_client import utils as client_utils
from modules.models.sequences.llm.base_model import BaseModel


USE_AUDIO_IN_VIDEO = True
CACHE_AUDIO_PATH = "storage/multimodals/Qwen2.5-Omni"


def format_history(history: list):
    messages = []
    for item in history:
        if isinstance(item["content"], str) and len(item["content"].strip()) > 0:
            messages.append({"role": item['role'], "content": item['content']})
        elif item["role"] == "user" and (isinstance(item["content"], list) or
                                         isinstance(item["content"], tuple)):
            file_path = item["content"][0]

            mime_type = client_utils.get_mimetype(file_path)
            if mime_type.startswith("image"):
                messages.append({
                    "role":
                        item['role'],
                    "content": [{
                        "type": "image",
                        "image": file_path
                    }]
                })
            elif mime_type.startswith("video"):
                convert_webm_to_mp4(file_path, file_path.replace(".webm", ".mp4"))
                messages.append({
                    "role":
                        item['role'],
                    "content": [{
                        "type": "video",
                        "video": file_path
                    }]
                })
            elif mime_type.startswith("audio"):
                messages.append({
                    "role":
                        item['role'],
                    "content": [{
                        "type": "audio",
                        "audio": file_path,
                    }]
                })
    return messages


def convert_webm_to_mp4(input_file, output_file):
    try:
        (
            ffmpeg
                .input(input_file)
                .output(output_file, acodec='aac', ar='16000', audio_bitrate='192k')
                .run(quiet=True, overwrite_output=True)
        )
        print(f"Conversion successful: {output_file}")
    except ffmpeg.Error as e:
        print("An error occurred during conversion.")
        print(e.stderr.decode('utf-8'))


class QwenOmiTransformerModel(BaseModel):
    def load_model(self, model_path: str, device: str):
        model = Qwen2_5OmniModel.from_pretrained(
            model_path,
            device_map=device.lower(),
            torch_dtype="auto",
            enable_audio_output=True,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        processor = Qwen2_5OmniProcessor.from_pretrained(
            model_path,
            device_map=device.lower(),
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        self.model = model
        self.processor = processor

    def generate_prompt(self, instruction: str):
        return f"""
                {instruction}
                """

    def chat(self, history, max_tokens, temperature, top_p, slider_context_times, return_audio=True):
        messages = [
            {"role": "system", "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."},
        ]

        format_messages = format_history(history)
        messages = messages + format_messages
        print("messages = ", messages)
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        audios, images, videos = process_mm_info(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        input_ids = self.processor(
            text=text,
            audios=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=USE_AUDIO_IN_VIDEO
        ).to(self.model.device).to(self.model.dtype)

        start_time = time.time()
        audio = None
        print('[Transformer] Human:', history[-1]["content"])
        print('[Transformer] Assistant: ', end='', flush=True)
        if return_audio:
            text_ids, audio = self.model.generate(
                **input_ids,
                spk="Chelsie",
                use_audio_in_video=USE_AUDIO_IN_VIDEO,
                return_audio=True
            )
        else:
            text_ids = self.model.generate(
                **input_ids,
                use_audio_in_video=USE_AUDIO_IN_VIDEO,
                return_audio=False
            )
        response = self.processor.batch_decode(
            text_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        bot_message = response[0].split("\n")[-1]
        print(bot_message, end='', flush=True)

        end_time = time.time()
        cost_time = round(end_time-start_time, 3)
        words_count = len(bot_message)
        single_word_cost_time = round((end_time-start_time)/len(bot_message), 3)
        per_second_tokens = round(len(text_ids[0]) / (end_time-start_time), 3)
        yield bot_message, cost_time, words_count, single_word_cost_time, per_second_tokens

        if return_audio:
            audio_path = os.path.join(CACHE_AUDIO_PATH, f"qwen_omni_7B_{int(time.time() * 1000)}.wav")
            sf.write(
                audio_path,
                audio.reshape(-1).detach().cpu().numpy(),
                samplerate=24000,
            )
            yield {"type": "audio", "data": audio_path}, 0, 0, 0, 0

    def release(self):
        del self.model
        del self.processor





