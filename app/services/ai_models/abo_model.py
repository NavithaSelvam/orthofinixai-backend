import cv2
import numpy as np
from .base_model import OrthodonticModel

class ABOScoringModel(OrthodonticModel):
    def __init__(self, model_path: str = None):
        self.model = None
        print("ABOScoringModel initialized with advanced Computer Vision logic.")

    def load_model(self, model_path: str):
        pass

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return image

    def predict(self, input_tensor: np.ndarray) -> dict:
        """
        Analyzes the image mathematically to score ABO OGS criteria (Alignment, Marginal Ridges, etc.).
        """
        gray = cv2.cvtColor(input_tensor, cv2.COLOR_BGR2GRAY)
        
        # 1. Isolate tooth boundaries using adaptive thresholding
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 2. Extract geometrical features (Contours)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for plausible tooth contours based on area
        img_area = input_tensor.shape[0] * input_tensor.shape[1]
        tooth_contours = [c for c in contours if (img_area * 0.005) < cv2.contourArea(c) < (img_area * 0.08)]
        
        if not tooth_contours:
            return {"score": 20, "details": "Image too blurry. Estimating default score."}
            
        # Calculate centers
        centers_y = []
        for c in tooth_contours:
            M = cv2.moments(c)
            if M['m00'] != 0:
                centers_y.append(M['m01']/M['m00'])
                
        # 3. ABO Criteria Calculation
        
        # A. Alignment (Variance of vertical position proxying rotation/misalignment)
        if len(centers_y) > 1:
            alignment_variance = np.std(centers_y)
        else:
            alignment_variance = 0
            
        alignment_penalty = min(15.0, alignment_variance / 10.0)
        
        # B. Marginal Ridge Discrepancy
        # Look at height differences between adjacent bounding boxes
        tooth_contours_sorted = sorted(tooth_contours, key=lambda c: cv2.boundingRect(c)[0])
        ridge_penalties = 0.0
        for i in range(len(tooth_contours_sorted)-1):
            _, y1, _, h1 = cv2.boundingRect(tooth_contours_sorted[i])
            _, y2, _, h2 = cv2.boundingRect(tooth_contours_sorted[i+1])
            diff = abs(y1 - y2)
            if diff > (input_tensor.shape[0] * 0.02): # > 2% vertical discrepancy
                ridge_penalties += 1.0
                
        ridge_penalty = min(8.0, ridge_penalties * 0.5)
        
        # C. Interproximal Contacts (Gaps)
        gaps = 0
        for i in range(len(tooth_contours_sorted)-1):
            x1, _, w1, _ = cv2.boundingRect(tooth_contours_sorted[i])
            x2, _, _, _ = cv2.boundingRect(tooth_contours_sorted[i+1])
            gap = x2 - (x1 + w1)
            if gap > (input_tensor.shape[1] * 0.015): # > 1.5% width gap
                gaps += 1.0
                
        gap_penalty = min(6.0, gaps * 1.5)

        # Base minimum score
        base_score = 5.0 
        final_score = round(base_score + alignment_penalty + ridge_penalty + gap_penalty, 1)
        
        return {
            "score": final_score,
            "details": f"Calculated dynamically via CV. Alignment Penalty: {round(alignment_penalty, 1)}, Marginal Ridges: {round(ridge_penalty, 1)}, Contacts: {round(gap_penalty, 1)}."
        }
