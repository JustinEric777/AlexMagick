import abc
from typing import Any


class BaseModel(metaclass=abc.ABCMeta):
    """
    Abstract base class for all models.
    Provides common attributes and the load_model interface.
    """
    model: Any = None
    device: Any = None
    tokenizer: Any = None
    processor: Any = None
    
    @abc.abstractmethod
    def load_model(self, model_path: str, device: str, **kwargs):
        """
        Load the model from the specified path.
        
        Args:
            model_path (str): Path to the model or model identifier.
            device (str): Device to load the model onto (e.g., "cpu", "cuda").
            **kwargs: Additional arguments for model loading.
        """
        pass
