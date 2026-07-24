import cv2
import numpy as np

class ImagePreprocessor:
    """
    Handles clinical image preprocessing before feeding into AI models.
    Supports standardizing OPG (Orthopantomogram) and intraoral photos.
    """
    
    @staticmethod
    def standardize_image(image: np.ndarray, target_size=(224, 224)) -> np.ndarray:
        """
        Resize image and normalize pixel values to [0, 1].
        """
        img_resized = cv2.resize(image, target_size)
        img_normalized = img_resized.astype(np.float32) / 255.0
        return img_normalized
        
    @staticmethod
    def enhance_contrast_clahe(image: np.ndarray) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
        Excellent for enhancing dental x-rays (OPG/Cephalometric) to reveal root/bone boundaries.
        """
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L-channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            
            # Merge and convert back
            limg = cv2.merge((cl,a,b))
            enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            return enhanced_img
        else:
            # Grayscale image (e.g., standard X-ray)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            return clahe.apply(image)

    @staticmethod
    def extract_edges_canny(image: np.ndarray) -> np.ndarray:
        """
        Extracts edges using Canny edge detection. Useful as an additional channel 
        for landmark detection models to focus on tooth boundaries.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return edges

    @classmethod
    def prepare_for_inference(cls, image: np.ndarray, model_type="classification") -> np.ndarray:
        """
        Full pipeline to prepare an image for a specific model type.
        """
        if model_type == "xray_segmentation":
            enhanced = cls.enhance_contrast_clahe(image)
            standardized = cls.standardize_image(enhanced, target_size=(512, 512))
            return np.expand_dims(standardized, axis=0) # Add batch dimension
            
        # Default classification preparation
        standardized = cls.standardize_image(image)
        return np.expand_dims(standardized, axis=0)
