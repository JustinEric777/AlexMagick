from modelscope.models import Model
from modelscope.preprocessors import Preprocessor
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modules.models.sequences.mt.base_model import BaseModel


class AliCSANMTModel(BaseModel):
    def load_model(self, model_path: str):
        model = Model.from_pretrained(model_path)
        processor = Preprocessor.from_pretrained(model_path)
        pipeline_mt = pipeline(task=Tasks.translation, model=model, preprocessor=processor)

        self.model = pipeline_mt
        self.processor = processor

    def translate(self, text: str):
        outputs = self.model(input=text)

        return outputs["translation"]
