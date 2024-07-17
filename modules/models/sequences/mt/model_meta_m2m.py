from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from modules.models.sequences.mt.base_model import BaseModel


class MetaM2MModel(BaseModel):
    def load_model(self, model_path: str):
        model = M2M100ForConditionalGeneration.from_pretrained(model_path)
        processor = M2M100Tokenizer.from_pretrained(model_path)

        self.model = model
        self.processor = processor

    def translate(self, text: str):
        self.processor.src_lang = "en"
        inputs = self.processor(text, return_tensors="pt")
        generated_ids = self.model.generate(
            **inputs,
            forced_bos_token_id=self.processor.get_lang_id("zh")
        )
        outputs = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return outputs[0]



