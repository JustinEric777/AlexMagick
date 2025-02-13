from transformers import T5ForConditionalGeneration, T5Tokenizer
from modules.models.sequences.mt.base_model import BaseModel


class GoogleT5Model(BaseModel):
    def load_model(self, model_path: str, device: str):
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        processor = T5Tokenizer.from_pretrained(model_path)

        self.model = model
        self.processor = processor

    def translate(self, text: str):
        text = f'translate to zh:{text}'
        inputs = self.processor(text, return_tensors="pt")
        generated_ids = self.model.generate(**inputs)
        outputs = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return outputs[0]

