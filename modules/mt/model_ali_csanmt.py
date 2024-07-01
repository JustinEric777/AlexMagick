from modelscope.models import Model
from modelscope.preprocessors import Preprocessor
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modules.mt.base_model import BaseModel


class AliCSANMTModel(BaseModel):
    def load_model(self, model_path: str):
        model = Model.from_pretrained(model_path)
        processor = Preprocessor.from_pretrained(model_path)

        self.model = model
        self.processor = processor

    def translate(self, text: str):
        pipeline_mt = pipeline(task=Tasks.translation, model=self.model, preprocessor=self.processor)
        outputs = pipeline_mt(input=text)

        return outputs["translation"]
