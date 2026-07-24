import math
from typing import Dict, Tuple, List, Any
from .geometry import calculate_angle_between_vectors

class OPGUprightingAnalyzer:
    """
    Analyzes panoramic radiographs (OPGs) to evaluate root parallelism,
    adjacent root deviations, and generate clinical uprighting recommendations.
    """
    
    @staticmethod
    def calculate_root_angulation(
        apex: Tuple[float, float], 
        crown: Tuple[float, float], 
        v_op_norm: Tuple[float, float],
        is_upper: bool
    ) -> float:
        """
        Calculates root angulation in degrees relative to the perpendicular to the occlusal plane.
        Formula:
        theta = arctan((Ya - Yc)/(Xa - Xc)) - 90 degrees (adjusted for occlusal plane reference)
        """
        xa, ya = apex
        xc, yc = crown
        
        # Calculate dy and dx
        dy = ya - yc
        dx = xa - xc
        
        if dx == 0:
            angle_rad = math.pi / 2 if dy > 0 else -math.pi / 2
        else:
            angle_rad = math.atan2(dy, dx)
            
        angle_deg = math.degrees(angle_rad)
        
        # Adjust based on jaw to get deviation relative to the normal of the occlusal plane
        # For upper teeth, normal vector points upwards (-Y in image coords, i.e., -90 deg)
        # For lower teeth, normal vector points downwards (+Y in image coords, i.e., +90 deg)
        if is_upper:
            # We want to measure tilt relative to -90 degrees
            deviation = angle_deg + 90.0
        else:
            # We want to measure tilt relative to +90 degrees
            deviation = angle_deg - 90.0
            
        # Normalize deviation to [-180, 180]
        while deviation > 180.0:
            deviation -= 360.0
        while deviation < -180.0:
            deviation += 360.0
            
        # Clinical adjustment: we want to represent mesial/distal tipping.
        # Let's return the deviation in degrees (positive means mesial tipping, negative means distal tipping depending on quadrant)
        return round(deviation, 2)

    @staticmethod
    def analyze_parallelism(
        landmarks: Dict[str, Tuple[float, float]],
        v_op_norm: Tuple[float, float],
        scale_factor: float
    ) -> Dict[str, Any]:
        """
        Calculates root angulation for all present teeth and compares adjacent roots.
        """
        angulations = {}
        
        # Group teeth by jaw
        upper_teeth = []
        lower_teeth = []
        
        # Parse landmarks to extract apex and midpoint for each tooth
        for key in landmarks.keys():
            if key.endswith("_apex"):
                fdi_str = key.split("_")[0]
                try:
                    fdi = int(fdi_str)
                except ValueError:
                    continue
                
                mid_key = f"{fdi}_midpoint"
                if mid_key in landmarks:
                    apex = landmarks[key]
                    crown = landmarks[mid_key]
                    is_upper = (fdi < 30)
                    
                    ang = OPGUprightingAnalyzer.calculate_root_angulation(
                        apex, crown, v_op_norm, is_upper
                    )
                    
                    angulations[fdi] = {
                        "angulation": ang,
                        "is_upper": is_upper,
                        "apex": apex,
                        "crown": crown
                    }
                    
                    if is_upper:
                        upper_teeth.append(fdi)
                    else:
                        lower_teeth.append(fdi)
                        
        # Sort teeth in anatomical sequence from right to left
        # FDI Maxillary: 18 -> 11, then 21 -> 28
        # We can sort upper teeth: quadrant 1 descending, quadrant 2 ascending
        q1_upper = sorted([t for t in upper_teeth if t < 20], reverse=True)
        q2_upper = sorted([t for t in upper_teeth if t >= 20])
        sorted_upper = q1_upper + q2_upper
        
        # FDI Mandibular: 48 -> 41, then 31 -> 38
        q4_lower = sorted([t for t in lower_teeth if t >= 40], reverse=True)
        q3_lower = sorted([t for t in lower_teeth if t < 40])
        sorted_lower = q4_lower + q3_lower
        
        deviations = []
        parallelism_score_sum = 0.0
        comparisons_count = 0
        
        # Compare adjacent upper roots
        for i in range(len(sorted_upper) - 1):
            t1, t2 = sorted_upper[i], sorted_upper[i+1]
            ang1 = angulations[t1]["angulation"]
            ang2 = angulations[t2]["angulation"]
            dev = abs(ang1 - ang2)
            
            status = "Parallel" if dev <= 5.0 else "Divergent" if (ang1 * ang2 < 0) else "Convergent"
            severity = "Normal" if dev <= 5.0 else "Mild" if dev <= 8.0 else "Moderate" if dev <= 12.0 else "Severe"
            
            deviations.append({
                "teeth": (t1, t2),
                "angulation_1": ang1,
                "angulation_2": ang2,
                "deviation_angle": round(dev, 2),
                "status": status,
                "severity": severity,
                "jaw": "upper"
            })
            
            # Penalize overall score for non-parallelism
            parallelism_score_sum += max(0.0, 1.0 - (dev / 15.0))
            comparisons_count += 1
            
        # Compare adjacent lower roots
        for i in range(len(sorted_lower) - 1):
            t1, t2 = sorted_lower[i], sorted_lower[i+1]
            ang1 = angulations[t1]["angulation"]
            ang2 = angulations[t2]["angulation"]
            dev = abs(ang1 - ang2)
            
            status = "Parallel" if dev <= 5.0 else "Divergent" if (ang1 * ang2 < 0) else "Convergent"
            severity = "Normal" if dev <= 5.0 else "Mild" if dev <= 8.0 else "Moderate" if dev <= 12.0 else "Severe"
            
            deviations.append({
                "teeth": (t1, t2),
                "angulation_1": ang1,
                "angulation_2": ang2,
                "deviation_angle": round(dev, 2),
                "status": status,
                "severity": severity,
                "jaw": "lower"
            })
            
            parallelism_score_sum += max(0.0, 1.0 - (dev / 15.0))
            comparisons_count += 1
            
        # Generate score
        overall_score = (parallelism_score_sum / max(1, comparisons_count)) * 100
        
        # Recommendations
        recommendations = []
        for dev_record in deviations:
            if dev_record["severity"] in ["Moderate", "Severe"]:
                t1, t2 = dev_record["teeth"]
                dev_val = dev_record["deviation_angle"]
                jaw_str = dev_record["jaw"]
                rec_text = (
                    f"Roots of adjacent teeth {t1} and {t2} in the {jaw_str} arch are non-parallel "
                    f"(deviation {dev_val}°). "
                )
                if dev_record["status"] == "Divergent":
                    rec_text += f"Apply artistic root-uprighting tipping bends mesially on {t1} or distally on {t2}."
                else:
                    rec_text += f"Apply tip-back adjustments to correct root convergence."
                    
                recommendations.append(rec_text)
                
        if not recommendations:
            recommendations.append("Excellent root parallelism observed. No tipping or uprighting adjustments needed.")
            
        return {
            "root_parallelism_score": round(overall_score, 1),
            "angulations": {k: v["angulation"] for k, v in angulations.items()},
            "deviations": deviations,
            "uprighting_recommendations": recommendations
        }
