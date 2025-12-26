import os.path
import time
import torch
import torchaudio
import ChatTTS
from core.models.audios.tts.base_model import BaseModel, AUDIO_PATH
from transformers.cache_utils import Cache
from typing import Optional, Tuple

# torch._dynamo.config.cache_size_limit = 64
# torch._dynamo.config.suppress_errors = True
# torch.set_float32_matmul_precision('high')

# Monkey Patch for ChatTTS issue with transformers > 4.48
# RuntimeError: narrow(): length must be non-negative.
def _prepare_generation_inputs_fixed(
    self,
    input_ids: torch.Tensor,
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    cache_position: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    use_cache=True,
):
    # With static cache, the `past_key_values` is None
    # TODO joao: standardize interface for the different Cache classes and remove of this if
    has_static_cache = False
    if past_key_values is None:
        if hasattr(self.gpt.layers[0], "self_attn"):
            past_key_values = getattr(
                self.gpt.layers[0].self_attn, "past_key_value", None
            )
        has_static_cache = past_key_values is not None

    past_length = 0
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            past_length = (
                int(cache_position[0])
                if cache_position is not None
                else past_key_values.get_seq_length()
            )
            try:
                max_cache_length = past_key_values.get_max_cache_shape()
            except:
                max_cache_length = (
                    past_key_values.get_max_length()
                )  # deprecated in transformers 4.48
            
            # FIX: Handle negative max_cache_length
            if max_cache_length is not None and max_cache_length < 0:
                max_cache_length = None

            cache_length = (
                past_length
                if max_cache_length is None
                else min(max_cache_length, past_length)
            )
        # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
        else:
            cache_length = past_length = past_key_values[0][0].shape[2]
            max_cache_length = None

        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if (
            attention_mask is not None
            and attention_mask.shape[1] > input_ids.shape[1]
        ):
            start = attention_mask.shape[1] - past_length
            input_ids = input_ids.narrow(1, -start, start)
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids.narrow(
                1, past_length, input_ids.size(1) - past_length
            )
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask.narrow(
                1, -max_cache_length, max_cache_length
            )

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask.eq(0), 1)
        if past_key_values:
            position_ids = position_ids.narrow(
                1, -input_ids.shape[1], input_ids.shape[1]
            )

    input_length = (
        position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
    )
    if cache_position is None:
        cache_position = torch.arange(
            past_length, past_length + input_length, device=input_ids.device
        )
    else:
        cache_position = cache_position.narrow(0, -input_length, input_length)

    if has_static_cache:
        past_key_values = None

    model_inputs = self._GenerationInputs(
        position_ids=position_ids,
        cache_position=cache_position,
        use_cache=use_cache,
    )

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs.inputs_embeds = inputs_embeds
    else:
        # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
        # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
        # TODO: use `next_tokens` directly instead.
        model_inputs.input_ids = input_ids.contiguous()

    model_inputs.past_key_values = past_key_values
    model_inputs.attention_mask = attention_mask

    return model_inputs

# Apply monkey patch
ChatTTS.model.gpt.GPT._prepare_generation_inputs = _prepare_generation_inputs_fixed


class ChatTTSModel(BaseModel):
    def load_model(self, model_path: str, device: str):
        model = ChatTTS.Chat()
        model.load(
            source="local",
            force_redownload=False,
            custom_path=model_path
        )

        self.device = device.lower()
        self.model = model

    def inference(self, texts: [], sample_wav: str = None):
        # sample_audio = torchaudio.load(sample_wav)
        # speaker = self.model.sample_audio_speaker(sample_audio)
        speaker = self.model.sample_random_speaker()
        params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_emb=speaker,
            # temperature=0.3,
            # top_P=0.1, # 降低 top_P
            # top_K=5, # 极大降低 top_K
        )

        # 显式指定参数，避免默认值引发问题
        params_refine_text = ChatTTS.Chat.RefineTextParams(
            prompt='[oral_2][laugh_0][break_6]',
            # top_P=0.7, 
            # top_K=20, 
            # temperature=0.7,
            # repetition_penalty=1.05,
        )

        wavs = self.model.infer(
            texts,
            params_refine_text=params_refine_text,
            params_infer_code=params_infer_code,
            use_decoder=True, # 确保解码器启用
        )

        if not os.path.exists(AUDIO_PATH):
            os.makedirs(AUDIO_PATH, exist_ok=True)

        base_audio_path = os.path.join(AUDIO_PATH, f"chattts_{int(time.time() * 1000)}")
        saved_paths = []
        for i in range(len(wavs)):
            current_audio_path = f"{base_audio_path}_{i}.wav"
            try:
                torchaudio.save(current_audio_path, torch.from_numpy(wavs[i]).unsqueeze(0), 24000)
            except:
                torchaudio.save(current_audio_path, torch.from_numpy(wavs[i]), 24000)
            saved_paths.append(current_audio_path)

        # Return the first path to maintain compatibility with existing UI that expects a single string
        # TODO: Update UI to handle list of paths if multiple texts are supported
        return saved_paths[0] if saved_paths else ""

    def release(self):
        del self.model
