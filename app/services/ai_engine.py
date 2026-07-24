import numpy as np
import cv2
import math
from typing import Dict, List, Tuple, Any, Optional

from app.services.ai_models.clinical_analysis.geometry import fit_occlusal_plane
from app.services.ai_models.clinical_analysis.segmentation import ToothSegmentationEngine
from app.services.ai_models.clinical_analysis.landmarks import LandmarkDetectionEngine
from app.services.ai_models.clinical_analysis.opg_uprighting import OPGUprightingAnalyzer
from app.services.ai_models.clinical_analysis.overjet_overbite import OverjetOverbiteAnalyzer
from app.services.ai_models.clinical_analysis.andrews_keys import AndrewsSixKeysAnalyzer

class OrthodonticAIEngine:
    """
    Main orchestrator of the OrthofinixAI clinical calculation pipeline.
    Combines tooth segmentation, landmark extraction, calibration, clinical rule-based scoring,
    conflict validation, and manual recalculation services.
    """
    def __init__(self):
        self.segmentation_engine = ToothSegmentationEngine()
        self.landmark_engine = LandmarkDetectionEngine()
        print("Clinically Accurate Orthodontic AI Engine Initialized.")

    def analyze_image(self, image_bytes: bytes, view_type: str = "frontal", bracket_pixel_width: float = 30.0) -> dict:
        """
        Main pipeline running preprocessing, segmentation, landmark detection, scaling,
        clinical rules evaluation, and conflict checks.
        """
        # 1. Decode and preprocess image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image format.")
            
        h, w = img.shape[:2]
        
        # 2. Run tooth instance segmentation (FDI numbered masks)
        segmented_teeth = self.segmentation_engine.segment_image(img, view_type=view_type)
        
        # Image Validation: Reject if no teeth/dental structures detected
        if len(segmented_teeth) == 0:
            raise ValueError("Please upload a valid dental image.")
        
        # 3. Detect anatomical landmarks anchored to segments
        landmarks = self.landmark_engine.detect_landmarks(img, segmented_teeth, view_type=view_type)
        
        # 4. Perform scale calibration (mm per normalized unit)
        # Standard width of bracket = 3.2 mm.
        # If bracket_pixel_width is 30 pixels in an 800px width image, normalized width is 30/800.
        # Scale = 3.2 / normalized_width
        norm_bracket_width = bracket_pixel_width / max(1.0, float(w))
        scale_factor = 3.2 / norm_bracket_width if norm_bracket_width > 0 else 85.0
        
        # 5. Fit Occlusal Plane (OP)
        # Fit OP using molar contacts and incisor edges
        op_points = []
        for key in ["16_midpoint", "26_midpoint", "36_midpoint", "46_midpoint", "11_incisal_edge", "21_incisal_edge"]:
            if key in landmarks:
                op_points.append(landmarks[key])
        if len(op_points) < 2:
            # Fallback points dynamically spread to avoid identical static scores on failure
            op_points = [(w * 0.2, h * 0.5), (w * 0.8, h * 0.5)]
            
        (slope, intercept), v_op_norm = fit_occlusal_plane(op_points)
        
        # 6. Execute Diagnostic Modules based on View Type
        details = {
            "view_type": view_type,
            "scale_factor": round(scale_factor, 2),
            "occlusal_plane": {"slope": round(slope, 4), "intercept": round(intercept, 4), "vector": [round(v_op_norm[0], 4), round(v_op_norm[1], 4)]},
            "detected_landmarks": landmarks,
            "segmented_teeth": segmented_teeth,
            "warnings": [],
            "conflicts": []
        }
        
        # OPG Root Uprighting
        opg_results = OPGUprightingAnalyzer.analyze_parallelism(landmarks, v_op_norm, scale_factor)
        details["root_parallelism"] = opg_results
        
        # Lateral Overjet and Overbite
        lateral_results = OverjetOverbiteAnalyzer.analyze_lateral_incisors(landmarks, v_op_norm, scale_factor)
        details["overjet_overbite"] = lateral_results
        
        # Andrews Six Keys & Molar Classification
        andrews_results = AndrewsSixKeysAnalyzer.run_full_analysis(landmarks, segmented_teeth, v_op_norm, scale_factor)
        details["andrews_details"] = andrews_results["details"]
        
        # 7. False Detection Reduction & Confidence Scoring
        confidence_score, warnings = self._validate_and_score_confidence(img, segmented_teeth, landmarks, view_type)
        details["warnings"].extend(warnings)
        
        # Conflict validation
        conflicts = self._validate_rules_conflicts(lateral_results, andrews_results)
        details["conflicts"].extend(conflicts)
        
        # Overall scoring aggregation
        andrews_score = andrews_results["overall_andrews_score"]
        root_angulation_score = opg_results["root_parallelism_score"]
        
        # Arch symmetry proxy (based on bilateral tooth centroids matching)
        arch_symmetry = self._calculate_arch_symmetry(segmented_teeth)
        
        # ABO finished scoring estimate (ideal is 0, penalty based)
        abo_penalty = self._calculate_abo_penalties(andrews_results, lateral_results, opg_results)
        
        # Formulate clinical recommendations
        recommendations = self._formulate_recommendations(andrews_results, lateral_results, opg_results, conflicts, view_type)
        
        return {
            "prediction": f"Analysis complete for {view_type.upper()} view.",
            "confidence_score": round(confidence_score, 2),
            "abo_score": round(abo_penalty, 1),
            "arch_symmetry_score": round(arch_symmetry, 1),
            "root_angulation_score": round(root_angulation_score, 1),
            "andrews_score": round(andrews_score, 1),
            "recommendations": recommendations,
            "details": details
        }

    def recalculate_from_landmarks(
        self, 
        landmarks: Dict[str, Tuple[float, float]], 
        segmented_teeth: Optional[Dict[int, Dict[str, Any]]] = None,
        view_type: str = "frontal", 
        bracket_pixel_width: float = 30.0,
        scale_factor: Optional[float] = None
    ) -> dict:
        """
        Recalculates clinical metrics based on user manual adjustment of landmarks.
        """
        if scale_factor is None:
            # Recalculate scale factor
            scale_factor = 3.2 / (bracket_pixel_width / 800.0) # standard width assumed
            
        # Fit Occlusal Plane
        op_points = []
        for key in ["16_midpoint", "26_midpoint", "36_midpoint", "46_midpoint", "11_incisal_edge", "21_incisal_edge"]:
            if key in landmarks:
                op_points.append(landmarks[key])
        if len(op_points) < 2:
            op_points = [(0.2, 0.5), (0.8, 0.5)]
            
        (slope, intercept), v_op_norm = fit_occlusal_plane(op_points)
        
        # Ensure segmented_teeth is not None
        if not segmented_teeth:
            segmented_teeth = {}
            
        # Recalculate
        opg_results = OPGUprightingAnalyzer.analyze_parallelism(landmarks, v_op_norm, scale_factor)
        lateral_results = OverjetOverbiteAnalyzer.analyze_lateral_incisors(landmarks, v_op_norm, scale_factor)
        andrews_results = AndrewsSixKeysAnalyzer.run_full_analysis(landmarks, segmented_teeth, v_op_norm, scale_factor)
        
        # Re-verify conflicts and warnings
        conflicts = self._validate_rules_conflicts(lateral_results, andrews_results)
        
        andrews_score = andrews_results["overall_andrews_score"]
        root_angulation_score = opg_results["root_parallelism_score"]
        arch_symmetry = self._calculate_arch_symmetry(segmented_teeth)
        abo_penalty = self._calculate_abo_penalties(andrews_results, lateral_results, opg_results)
        recommendations = self._formulate_recommendations(andrews_results, lateral_results, opg_results, conflicts, view_type)
        
        return {
            "abo_score": round(abo_penalty, 1),
            "arch_symmetry_score": round(arch_symmetry, 1),
            "root_angulation_score": round(root_angulation_score, 1),
            "andrews_score": round(andrews_score, 1),
            "recommendations": recommendations,
            "details": {
                "view_type": view_type,
                "scale_factor": round(scale_factor, 2),
                "occlusal_plane": {"slope": round(slope, 4), "intercept": round(intercept, 4), "vector": [round(v_op_norm[0], 4), round(v_op_norm[1], 4)]},
                "detected_landmarks": landmarks,
                "segmented_teeth": segmented_teeth,
                "root_parallelism": opg_results,
                "overjet_overbite": lateral_results,
                "andrews_details": andrews_results["details"],
                "conflicts": conflicts
            }
        }

    def _validate_and_score_confidence(self, image: np.ndarray, segmented_teeth: Dict[int, Dict[str, Any]], landmarks: Dict[str, Tuple[float, float]], view_type: str) -> Tuple[float, List[str]]:
        """
        False Detection Reduction: anatomy constraints, edge clarity, and view consistency.
        """
        warnings = []
        base_confidence = 0.95
        
        # 1. Image quality (blur check)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_val < 80:
            base_confidence -= 0.15
            warnings.append("Image appears blurry or low contrast. Landmark accuracy might be compromised.")
            
        # 2. Anatomy check: upper centroids must be above lower centroids
        upper_y = [t["centroid"][1] for fdi, t in segmented_teeth.items() if fdi < 30]
        lower_y = [t["centroid"][1] for fdi, t in segmented_teeth.items() if fdi >= 30]
        
        if upper_y and lower_y:
            avg_upper_y = sum(upper_y) / len(upper_y)
            avg_lower_y = sum(lower_y) / len(lower_y)
            
            if avg_upper_y >= avg_lower_y:
                base_confidence -= 0.3
                warnings.append("Anatomical conflict: Maxillary teeth centroids detected below mandibular teeth.")
                
        # 3. Tooth count check based on view type
        expected_min = 2 if view_type == "lateral" else 8
        if len(segmented_teeth) < expected_min:
            base_confidence -= 0.2
            warnings.append(f"Fewer teeth segmented ({len(segmented_teeth)}) than expected for {view_type} view.")
            
        # Confidence gate warning
        if base_confidence < 0.75:
            warnings.append("Detection confidence low. Please verify landmarks manually.")
            
        return max(0.2, base_confidence), warnings

    def _validate_rules_conflicts(self, lateral: dict, andrews: dict) -> List[str]:
        """
        Rule conflict validation. Checks consistency across orthodontic rules.
        """
        conflicts = []
        
        # Molar relationship Class III vs Overjet Class II
        molar_details = next((k for k in andrews.get("details", []) if "Molar" in k.get("key", "")), {})
        molar_left = molar_details.get("details", {}).get("left", {}).get("classification")
        molar_right = molar_details.get("details", {}).get("right", {}).get("classification")
        
        overjet_val = lateral.get("overjet_mm", 2.0)
        
        if (molar_left == "Class II" or molar_right == "Class II") and overjet_val < 0.0:
            conflicts.append("Clinical conflict: Class II molar tendency detected alongside anterior crossbite (negative overjet). Please double-check molar landmarks.")
            
        if (molar_left == "Class III" or molar_right == "Class III") and overjet_val > 5.0:
            conflicts.append("Clinical conflict: Class III molar tendency detected alongside severe overjet (protrusion). Please double-check incisor position.")
            
        return conflicts

    def _calculate_arch_symmetry(self, segmented_teeth: Dict[int, Dict[str, Any]]) -> float:
        """Computes the bilateral horizontal symmetry of the dental arches."""
        left_centroids = []
        right_centroids = []
        
        for fdi, tooth in segmented_teeth.items():
            cx = tooth["centroid"][0]
            quad = fdi // 10
            if quad in [2, 3]: # Left side
                left_centroids.append(cx)
            elif quad in [1, 4]: # Right side
                right_centroids.append(cx)
                
        if not left_centroids or not right_centroids:
            return 90.0 # Default baseline
            
        # Average distance from central midline (assumed X = 0.5)
        avg_left_dist = abs(sum(left_centroids)/len(left_centroids) - 0.5)
        avg_right_dist = abs(sum(right_centroids)/len(right_centroids) - 0.5)
        
        diff = abs(avg_left_dist - avg_right_dist)
        symmetry = 100.0 - (diff * 200.0)
        return max(50.0, min(99.0, symmetry))

    def _calculate_abo_penalties(self, andrews: dict, lateral: dict, opg: dict) -> float:
        """
        Estimates the American Board of Orthodontics (ABO) Objective Grading System penalty.
        ABO OGS scores are penalties, where lower scores are better. Ideal = 0.
        """
        penalty = 0.0
        
        # 1. Alignment penalty
        rot_details = next((k for k in andrews.get("details", []) if "Rotations" in k.get("key", "")), {})
        penalty += len(rot_details.get("violations", [])) * 1.5
        
        # 2. Marginal ridge penalty
        contacts_details = next((k for k in andrews.get("details", []) if "Contacts" in k.get("key", "")), {})
        penalty += len(contacts_details.get("violations", [])) * 1.0
        
        # 3. Overjet / Overbite penalties
        oj_val = lateral.get("overjet_mm", 2.0)
        ob_percent = lateral.get("overbite_percent", 30.0)
        
        if oj_val > 4.0:
            penalty += int(oj_val - 4.0) * 2.0
        elif oj_val < 0.0:
            penalty += 4.0 # severe crossbite
            
        if ob_percent > 40.0:
            penalty += ((ob_percent - 40.0) / 10.0) * 1.5
        elif ob_percent < 0.0:
            penalty += 3.0
            
        # 4. Root parallelism penalty
        devs = opg.get("deviations", [])
        for dev in devs:
            if dev["severity"] == "Severe":
                penalty += 2.0
            elif dev["severity"] == "Moderate":
                penalty += 1.0
                
        return penalty

    def _formulate_recommendations(self, andrews: dict, lateral: dict, opg: dict, conflicts: list, view: str) -> List[str]:
        """
        Creates actionable, explainable clinical treatment recommendations based on rules violations.
        """
        recommendations = []
        
        # Andrews Keys violations
        for key_data in andrews.get("details", []):
            key_name = key_data.get("key", "")
            
            # Key 1 Molar
            if "Molar" in key_name and key_data["score"] < 0.9:
                details = key_data.get("details", {})
                for side in ["left", "right"]:
                    s_res = details.get(side, {})
                    if s_res.get("classification") in ["Class II", "Class III"]:
                        recommendations.append(
                            f"Key 1 ({side.capitalize()} Molar): {s_res['explanation']} "
                            f"Use Class {'II' if s_res['classification']=='Class II' else 'III'} elastics or molar distalization."
                        )
                        
            # Key 2 Angulation (Tip)
            elif "Angulation" in key_name and key_data["score"] < 0.9:
                violations = key_data.get("violations", [])
                for v in violations[:3]: # Limit to top 3 to keep UI tidy
                    recommendations.append(
                        f"Key 2 (Tip): {v['explanation']} "
                        f"Place an artistic detailing bend of {v['deviation']}° to upright the crown."
                    )
                    
            # Key 3 Inclination (Torque)
            elif "Inclination" in key_name and key_data["score"] < 0.9:
                violations = key_data.get("violations", [])
                for v in violations[:3]:
                    recommendations.append(
                        f"Key 3 (Torque): {v['explanation']} "
                        f"Adjust rectangular archwire ({v['ideal']}° ideal) to introduce proper third-order torque control."
                    )
                    
            # Key 4 Rotations
            elif "Rotations" in key_name and key_data["score"] < 0.9:
                violations = key_data.get("violations", [])
                for v in violations[:3]:
                    recommendations.append(
                        f"Key 4 (Rotations): {v['explanation']} "
                        f"Correct using rotational wedges, offset bracket placement, or a nitinol utility arch."
                    )
                    
            # Key 5 Contacts/Spacing
            elif "Contacts" in key_name and key_data["score"] < 0.9:
                violations = key_data.get("violations", [])
                for v in violations[:3]:
                    recommendations.append(
                        f"Key 5 (Contacts): {v['explanation']} "
                        f"Apply power chain elastics or open-coil springs to resolve the {v['deviation_mm']} mm {v['type'].lower()}."
                    )
                    
            # Key 6 Curve of Spee
            elif "Spee" in key_name and key_data["score"] < 0.9:
                recommendations.append(
                    f"Key 6 (Spee): {key_data['explanation']} "
                    "Use reverse-curve of Spee (RCS) archwires or anterior bite turbos to level the arch."
                )
                
        # Overjet / Overbite
        if view == "lateral":
            oj_status = lateral.get("overjet_status")
            ob_status = lateral.get("overbite_status")
            
            if oj_status == "Excessive Overjet":
                recommendations.append(f"Overjet is excessive. Level incisors and consider Class II intermaxillary elastics.")
            elif oj_status == "Anterior Crossbite / Underjet":
                recommendations.append("Anterior crossbite detected. Prioritize anterior bite opening and check for skeletal Class III growth patterns.")
                
            if ob_status == "Deep Bite":
                recommendations.append(f"Deep bite ({lateral['overbite_percent']}%). Intrudes mandibular incisors or utilizes anterior bite plates.")
            elif ob_status == "Anterior Open Bite":
                recommendations.append("Anterior open bite detected. Cease thumbsucking/tongue habit if active; check vertical skeletal dimensions.")
                
        # OPG parallelism
        if view == "opg":
            opg_recs = opg.get("uprighting_recommendations", [])
            for r in opg_recs:
                if "Excellent" not in r:
                    recommendations.append(r)
                    
        # Conflicts
        for c in conflicts:
            recommendations.append(f"Validation Warning: {c}")
            
        if not recommendations:
            recommendations.append("Occlusion is finished. Alignment meets all Andrews Six Keys criteria.")
            
        return recommendations

# Singleton instance
ai_engine = OrthodonticAIEngine()
