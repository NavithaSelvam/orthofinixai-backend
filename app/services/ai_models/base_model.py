from abc import ABC, abstractmethod
import numpy as np

class OrthodonticModel(ABC):
    """
    Abstract base class for all orthodontic AI models.
    Ensures a consistent interface for loading models and performing inference.
    """
    
    @abstractmethod
    def load_model(self, model_path: str):
        """
        Load the model weights from the given path (.h5, .tflite, .pt).
        """
        pass

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the OpenCV image (resize, normalize, expand dims) 
        before feeding it to the model.
        """
        pass

    @abstractmethod
    def predict(self, input_tensor: np.ndarray) -> dict:
        """
        Run inference and return the parsed results.
        """
        pass
