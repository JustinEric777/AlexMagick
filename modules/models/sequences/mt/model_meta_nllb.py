from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from modules.models.sequences.mt.base_model import BaseModel


class MetaNLLBModel(BaseModel):
    def load_model(self, model_path: str, device: str):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        processor = AutoTokenizer.from_pretrained(model_path)

        self.model = model
        self.processor = processor

    def translate(self, text: str):
        inputs = self.processor(text, return_tensors="pt")
        generated_ids = self.model.generate(
            **inputs,
            forced_bos_token_id=self.processor.lang_code_to_id["zho_Hans"] if hasattr(self.processor, "lang_code_to_id") else None,
            max_length=512
        )
        outputs = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return outputs





