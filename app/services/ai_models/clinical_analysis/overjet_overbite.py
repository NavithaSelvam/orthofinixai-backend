import math
from typing import Dict, Tuple, Any, Optional
from .geometry import project_vector_magnitude, calculate_distance

class OverjetOverbiteAnalyzer:
    """
    Computes Overjet (OJ) and Overbite (OB) measurements from a lateral clinical photo
    using vector projection onto the occlusal plane.
    """
    
    @staticmethod
    def analyze_lateral_incisors(
        landmarks: Dict[str, Tuple[float, float]],
        v_op_norm: Tuple[float, float],
        scale_factor: float
    ) -> Dict[str, Any]:
        """
        Calculates overjet in mm and overbite percentage.
        """
        # Resolve upper incisor edge (Ui)
        ui = landmarks.get("11_incisal_edge") or landmarks.get("21_incisal_edge")
        
        # Resolve lower incisor edge (Li)
        li = landmarks.get("41_incisal_edge") or landmarks.get("31_incisal_edge")
        
        # Resolve lower incisor CEJ (Lc)
        lc = landmarks.get("41_cej_mesial") or landmarks.get("31_cej_mesial")
        
        if not ui or not li or not lc:
            return {
                "overjet_mm": 0.0,
                "overbite_mm": 0.0,
                "overbite_percent": 0.0,
                "crown_height_mm": 0.0,
                "overjet_status": "Insufficient Data",
                "overjet_severity": "N/A",
                "overjet_explanation": "Unable to calculate overjet: Landmarks missing.",
                "overbite_status": "Insufficient Data",
                "overbite_severity": "N/A",
                "overbite_explanation": "Unable to calculate overbite: Landmarks missing."
            }
            
        # LL is the most labial point of the lower incisor.
        ll_offset_x = -0.015
        ll = (li[0] + ll_offset_x, li[1])
        
        ux, uy = v_op_norm
        n_op = (-uy, ux)
        n_op_len = math.sqrt(n_op[0]**2 + n_op[1]**2)
        n_op_norm = (n_op[0]/n_op_len, n_op[1]/n_op_len)
        
        v_oj = (ui[0] - ll[0], ui[1] - ll[1])
        oj_normalized = project_vector_magnitude(v_oj, v_op_norm)
        oj_mm = oj_normalized * scale_factor
        
        v_ob = (ui[0] - li[0], ui[1] - li[1])
        ob_normalized = project_vector_magnitude(v_ob, n_op_norm)
        ob_mm = abs(ob_normalized * scale_factor)
        if ui[1] > li[1]:
            ob_mm = -ob_mm
            
        crown_height_norm = calculate_distance(li, lc)
        crown_height_mm = crown_height_norm * scale_factor

        if crown_height_mm <= 0:
             return {
                "overjet_mm": round(oj_mm, 2),
                "overbite_mm": round(ob_mm, 2),
                "overbite_percent": 0.0,
                "crown_height_mm": 0.0,
                "overjet_status": "Incomplete",
                "overjet_severity": "N/A",
                "overjet_explanation": f"Overjet is {round(oj_mm, 1)} mm.",
                "overbite_status": "Incomplete",
                "overbite_severity": "N/A",
                "overbite_explanation": "Overbite percentage calculation failed: crown height invalid."
            }

        ob_percent = (ob_mm / crown_height_mm) * 100.0
        
        oj_status = "Normal"
        oj_severity = "Normal"
        oj_explanation = f"Overjet is {round(oj_mm, 1)} mm. "
        
        if 2.0 <= oj_mm <= 4.0:
            oj_status = "Normal Overjet"
        elif oj_mm > 4.0:
            oj_status = "Excessive Overjet"
            oj_severity = "Mild" if oj_mm <= 6.0 else "Moderate" if oj_mm <= 8.0 else "Severe"
            oj_explanation += "Indicates Class II malocclusion tendency."
        else:
            if oj_mm < 0.0:
                oj_status = "Anterior Crossbite / Underjet"
                oj_severity = "Severe"
                oj_explanation += "Class III skeletal or dental pattern."
            else:
                oj_status = "Edge-to-Edge / Reduced Overjet"
                oj_severity = "Mild"

        ob_status = "Normal"
        ob_severity = "Normal"
        ob_explanation = f"Overbite is {round(ob_percent, 1)}% ({round(ob_mm, 1)} mm). "
        
        if ob_mm < 0.0:
            ob_status = "Anterior Open Bite"
            ob_severity = "Severe"
        elif 20.0 <= ob_percent <= 40.0:
            ob_status = "Normal Overbite"
        elif ob_percent > 40.0:
            ob_status = "Deep Bite"
            ob_severity = "Mild" if ob_percent <= 60.0 else "Moderate" if ob_percent <= 80.0 else "Severe"
        else:
            ob_status = "Reduced Overbite"

        return {
            "overjet_mm": round(oj_mm, 2),
            "overbite_mm": round(ob_mm, 2),
            "overbite_percent": round(ob_percent, 1),
            "crown_height_mm": round(crown_height_mm, 2),
            "overjet_status": oj_status,
            "overjet_severity": oj_severity,
            "overjet_explanation": oj_explanation,
            "overbite_status": ob_status,
            "overbite_severity": ob_severity,
            "overbite_explanation": ob_explanation
        }
