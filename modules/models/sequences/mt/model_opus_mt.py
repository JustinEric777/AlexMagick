from transformers import AutoTokenizer, MarianMTModel
from modules.models.sequences.mt.base_model import BaseModel


class OpusMTModel(BaseModel):
    def load_model(self, model_path: str):
        model = MarianMTModel.from_pretrained(model_path)
        processor = AutoTokenizer.from_pretrained(model_path)

        self.model = model
        self.processor = processor

    def translate(self, text: str):
        inputs = self.processor([text], return_tensors="pt")
        generated_ids = self.model.generate(**inputs)
        outputs = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return outputs




