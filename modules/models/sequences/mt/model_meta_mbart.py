from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from modules.models.sequences.mt.base_model import BaseModel


class MetaMBartModel(BaseModel):
    def load_model(self, model_path: str, device: str):
        model = MBartForConditionalGeneration.from_pretrained(model_path)
        processor = MBart50TokenizerFast.from_pretrained(model_path)

        self.model = model
        self.processor = processor

    def translate(self, text: str):
        self.processor.src_lang = "en_XX"
        inputs = self.processor(text, return_tensors="pt")
        generated_ids = self.model.generate(
            **inputs,
            forced_bos_token_id=self.processor.lang_code_to_id["zh_CN"]
        )
        outputs = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return outputs[0]

    def release(self):
        del self.model
        del self.processor


