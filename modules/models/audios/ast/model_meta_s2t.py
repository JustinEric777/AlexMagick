from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from modules.models.audios.ast.base_model import BaseModel


class MetaS2TModel(BaseModel):
    def load_model(self, model_path: str, device: str):
        model = Speech2TextForConditionalGeneration.from_pretrained(model_path)
        processor = Speech2TextProcessor.from_pretrained(model_path)

        self.model = model
        self.processor = processor

    def generate(self, text: str):
        inputs = self.processor(text, sampling_rate=16_000, return_tensors="pt")
        generated_ids = self.model.generate(
            input_ids=inputs["input_features"],
            attention_mask=inputs["attention_mask"],
            forced_bos_token_id=self.processor.tokenizer.lang_code_to_id["de"]
        )
        outputs = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return outputs

