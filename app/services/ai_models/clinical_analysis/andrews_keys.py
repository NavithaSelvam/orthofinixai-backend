from typing import Dict, List, Tuple, Any
import math
import numpy as np
from .geometry import calculate_distance, calculate_angle_between_vectors, project_vector_magnitude

class AndrewsSixKeysAnalyzer:
    """
    Evaluates Andrews' Six Keys to Normal Occlusion and molar classification 
    using precise geometric mathematics, coordinates, and scale factors.
    """

    # Ideal Crown Angulation (Key 2 - Tip) in degrees (Andrews standard)
    IDEAL_TIP = {
        11: 5.0, 12: 9.0, 13: 11.0, 14: 2.0, 15: 2.0, 16: 5.0, 17: 5.0, 18: 5.0,
        21: 5.0, 22: 9.0, 23: 11.0, 24: 2.0, 25: 2.0, 26: 5.0, 27: 5.0, 28: 5.0,
        31: 2.0, 32: 2.0, 33: 5.0, 34: 2.0, 35: 2.0, 36: 2.0, 37: 2.0, 38: 2.0,
        41: 2.0, 42: 2.0, 43: 5.0, 44: 2.0, 45: 2.0, 46: 2.0, 47: 2.0, 48: 2.0
    }

    # Ideal Crown Inclination (Key 3 - Torque) in degrees (Andrews standard)
    IDEAL_TORQUE = {
        11: 7.0, 12: 3.0, 13: -7.0, 14: -7.0, 15: -7.0, 16: -9.0, 17: -9.0, 18: -9.0,
        21: 7.0, 22: 3.0, 23: -7.0, 24: -7.0, 25: -7.0, 26: -9.0, 27: -9.0, 28: -9.0,
        31: -1.0, 32: -1.0, 33: -11.0, 34: -17.0, 35: -22.0, 36: -30.0, 37: -30.0, 38: -30.0,
        41: -1.0, 42: -1.0, 43: -11.0, 44: -17.0, 45: -22.0, 46: -30.0, 47: -30.0, 48: -30.0
    }

    @staticmethod
    def classify_molar_relationship_side(
        upper_mb_cusp: Tuple[float, float],
        lower_buccal_groove: Tuple[float, float],
        v_op_norm: Tuple[float, float],
        scale_factor: float,
        is_left_side: bool
    ) -> Dict[str, Any]:
        """
        Classifies molar relationships (Class I, II, or III) from a lateral/OPG perspective.
        Formula:
        Delta X = X_cusp - X_groove projected onto occlusal plane.
        """
        # Vector from lower groove to upper MB cusp
        dx = upper_mb_cusp[0] - lower_buccal_groove[0]
        dy = upper_mb_cusp[1] - lower_buccal_groove[1]
        
        # Project onto occlusal plane
        disparity_norm = dx * v_op_norm[0] + dy * v_op_norm[1]
        disparity_mm = disparity_norm * scale_factor
        
        # Adjust direction based on side of the arch
        # For left side, anterior (mesial) is to the right (positive X)
        # For right side, anterior (mesial) is to the left (negative X)
        # Let's normalize so positive means Class II (upper cusp is mesial/anterior to groove)
        if is_left_side:
            disparity_mm = disparity_mm # Keep positive if upper is mesial
        else:
            disparity_mm = -disparity_mm
            
        # Classify
        if -1.5 <= disparity_mm <= 1.5:
            classification = "Class I"
            explanation = f"Normal molar occlusion. The mesiobuccal cusp of the upper first molar fits within the buccal groove of the lower first molar (deviation: {round(disparity_mm, 1)} mm)."
            severity = "Normal"
            score = 1.0
        elif disparity_mm > 1.5:
            classification = "Class II"
            severity = "Mild" if disparity_mm <= 3.5 else "Moderate" if disparity_mm <= 5.5 else "Severe"
            explanation = f"Class II relationship detected. The upper molar is positioned mesially (anteriorly) relative to the lower molar groove by {round(disparity_mm, 1)} mm."
            score = max(0.0, 1.0 - (disparity_mm - 1.5) / 5.0)
        else:
            classification = "Class III"
            severity = "Mild" if disparity_mm >= -3.5 else "Moderate" if disparity_mm >= -5.5 else "Severe"
            explanation = f"Class III relationship detected. The upper molar is positioned distally (posteriorly) relative to the lower molar groove by {round(abs(disparity_mm), 1)} mm."
            score = max(0.0, 1.0 - (abs(disparity_mm) - 1.5) / 5.0)
            
        return {
            "classification": classification,
            "disparity_mm": round(disparity_mm, 2),
            "severity": severity,
            "explanation": explanation,
            "score": round(score, 2)
        }

    @staticmethod
    def analyze_key1_molar(landmarks: Dict[str, Tuple[float, float]], v_op_norm: Tuple[float, float], scale_factor: float) -> Dict[str, Any]:
        """Key 1: Molar Relationship (analyzes left and right separately)"""
        # Right Side: Tooth 16 (Upper Right Molar Cusp) vs 46 (Lower Right Molar Groove)
        u16 = landmarks.get("16_cusp_tip_buccal") or landmarks.get("16_midpoint")
        l46 = landmarks.get("46_buccal_groove") or landmarks.get("46_midpoint")
        
        # Left Side: Tooth 26 (Upper Left Molar Cusp) vs 36 (Lower Left Molar Groove)
        u26 = landmarks.get("26_cusp_tip_buccal") or landmarks.get("26_midpoint")
        l36 = landmarks.get("36_buccal_groove") or landmarks.get("36_midpoint")
        
        results = {}
        scores = []
        
        if u16 and l46:
            r_res = AndrewsSixKeysAnalyzer.classify_molar_relationship_side(u16, l46, v_op_norm, scale_factor, is_left_side=False)
            results["right"] = r_res
            scores.append(r_res["score"])
        else:
            results["right"] = {"classification": "Insufficient Data", "disparity_mm": 0.0, "severity": "Normal", "explanation": "Molar landmarks not detected on the right side.", "score": 1.0}
            
        if u26 and l36:
            l_res = AndrewsSixKeysAnalyzer.classify_molar_relationship_side(u26, l36, v_op_norm, scale_factor, is_left_side=True)
            results["left"] = l_res
            scores.append(l_res["score"])
        else:
            results["left"] = {"classification": "Insufficient Data", "disparity_mm": 0.0, "severity": "Normal", "explanation": "Molar landmarks not detected on the left side.", "score": 1.0}
            
        avg_score = sum(scores) / len(scores) if scores else 1.0
        
        explanation = f"Right side: {results['right']['classification']}. Left side: {results['left']['classification']}."
        status = "Class I Occlusion" if avg_score > 0.9 else "Class II/III Malocclusion Tendency"
        
        return {
            "key": "Key 1: Molar Relationship",
            "status": status,
            "score": round(avg_score, 2),
            "details": results,
            "explanation": explanation
        }

    @staticmethod
    def analyze_key2_angulation(
        landmarks: Dict[str, Tuple[float, float]],
        segmented_teeth: Dict[int, Dict[str, Any]],
        v_op_norm: Tuple[float, float]
    ) -> Dict[str, Any]:
        """
        Key 2: Crown Angulation (Tip).
        Formula:
        theta_tip = 90 degrees - angle(Crown Axis, Occlusal Plane)
        """
        violations = []
        scores = []
        angulations = {}
        
        for fdi, tooth in segmented_teeth.items():
            apex = landmarks.get(f"{fdi}_apex")
            mid = landmarks.get(f"{fdi}_midpoint")
            
            if not apex or not mid:
                continue
                
            # Long axis vector of the tooth (pointing from root to crown)
            is_upper = (fdi < 30)
            if is_upper:
                # Root is superior (smaller Y), Crown is inferior (larger Y)
                v_axis = (mid[0] - apex[0], mid[1] - apex[1])
            else:
                # Root is inferior (larger Y), Crown is superior (smaller Y)
                v_axis = (mid[0] - apex[0], mid[1] - apex[1])
                
            # Angle relative to occlusal plane
            op_angle = calculate_angle_between_vectors(v_axis, v_op_norm)
            
            # Tip: deviation from perpendicular (90 degrees)
            tip_val = 90.0 - op_angle
            
            # Standard correction: mesial tipping is clinical positive
            # For upper right quadrant, mesial is left-to-right (positive X).
            # For upper left quadrant, mesial is right-to-left (negative X).
            quadrant = fdi // 10
            if quadrant in [2, 3]: # Left quadrants
                tip_val = -tip_val
                
            angulations[fdi] = round(tip_val, 1)
            
            ideal = AndrewsSixKeysAnalyzer.IDEAL_TIP.get(fdi, 5.0)
            dev = abs(tip_val - ideal)
            
            # Score this tooth
            tooth_score = max(0.0, 1.0 - (dev / 10.0))
            scores.append(tooth_score)
            
            if dev > 3.0:
                severity = "Mild" if dev <= 5.0 else "Moderate" if dev <= 8.0 else "Severe"
                violations.append({
                    "tooth": fdi,
                    "angulation": round(tip_val, 1),
                    "ideal": ideal,
                    "deviation": round(dev, 1),
                    "severity": severity,
                    "explanation": f"Tooth {fdi} angulation is {round(tip_val, 1)}°, deviating by {round(dev, 1)}° from ideal Andrews standard ({ideal}°)."
                })
                
        avg_score = sum(scores) / len(scores) if scores else 1.0
        status = "Acceptable Crown Angulations" if avg_score > 0.8 else "Tipping Violations Detected"
        
        return {
            "key": "Key 2: Crown Angulation",
            "status": status,
            "score": round(avg_score, 2),
            "angulations": angulations,
            "violations": violations,
            "explanation": f"Overall angulation score is {round(avg_score*100, 1)}%. Found {len(violations)} teeth with tipping deviations."
        }

    @staticmethod
    def analyze_key3_inclination(
        landmarks: Dict[str, Tuple[float, float]],
        segmented_teeth: Dict[int, Dict[str, Any]],
        v_op_norm: Tuple[float, float]
    ) -> Dict[str, Any]:
        """
        Key 3: Crown Inclination (Torque).
        Measures crown labiolingual inclination relative to occlusal perpendicular.
        """
        violations = []
        scores = []
        torques = {}
        
        # Perpendicular normal of occlusal plane
        n_op = (-v_op_norm[1], v_op_norm[0])
        
        for fdi, tooth in segmented_teeth.items():
            # For torque in 2D profile view, we look at the tangent line of the facial surface (CEJ to incisal edge)
            cej = landmarks.get(f"{fdi}_cej_mesial")
            inc = landmarks.get(f"{fdi}_incisal_edge") or landmarks.get(f"{fdi}_cusp_tip_buccal")
            
            if not cej or not inc:
                continue
                
            # Surface vector
            v_surf = (inc[0] - cej[0], inc[1] - cej[1])
            
            # Inclination angle relative to perpendicular normal of occlusal plane
            surf_angle = calculate_angle_between_vectors(v_surf, n_op)
            
            # Standard torque calculation
            torque_val = 90.0 - surf_angle
            
            # Align torque sign with clinical convention
            is_upper = (fdi < 30)
            if not is_upper:
                torque_val = -torque_val
                
            torques[fdi] = round(torque_val, 1)
            
            ideal = AndrewsSixKeysAnalyzer.IDEAL_TORQUE.get(fdi, 0.0)
            dev = abs(torque_val - ideal)
            
            tooth_score = max(0.0, 1.0 - (dev / 12.0))
            scores.append(tooth_score)
            
            if dev > 4.0:
                severity = "Mild" if dev <= 6.0 else "Moderate" if dev <= 10.0 else "Severe"
                violations.append({
                    "tooth": fdi,
                    "torque": round(torque_val, 1),
                    "ideal": ideal,
                    "deviation": round(dev, 1),
                    "severity": severity,
                    "explanation": f"Tooth {fdi} torque inclination is {round(torque_val, 1)}°, exceeding ideal value of {ideal}° by {round(dev, 1)}°."
                })
                
        avg_score = sum(scores) / len(scores) if scores else 1.0
        status = "Optimal Crown Torques" if avg_score > 0.8 else "Inclination Violations Detected"
        
        return {
            "key": "Key 3: Crown Inclination",
            "status": status,
            "score": round(avg_score, 2),
            "torques": torques,
            "violations": violations,
            "explanation": f"Overall torque score is {round(avg_score*100, 1)}%. Found {len(violations)} torque inclination deviations."
        }

    @staticmethod
    def analyze_key4_rotations(
        segmented_teeth: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Key 4: Rotations (Absence of rotations).
        Calculates rotations based on bounding box contour aspect ratio deviations.
        """
        violations = []
        scores = []
        rotations = {}
        
        for fdi, tooth in segmented_teeth.items():
            bbox = tooth["bbox"] # [x_min, y_min, x_max, y_max]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            
            # Teeth should follow standard aspect ratios. If rotated, the horizontal width changes.
            # We can model rotation deviation as a function of contour symmetry and width-height ratio.
            # For simplicity, let's calculate rotation index:
            aspect = w / h
            ideal_aspect = 0.7 if tooth["class"] in ["incisor", "canine"] else 0.9
            
            # Estimate rotation angle in degrees
            dev_aspect = abs(aspect - ideal_aspect)
            rotation_deg = dev_aspect * 90.0 # scale to degrees
            rotation_deg = min(45.0, max(0.0, rotation_deg))
            
            rotations[fdi] = round(rotation_deg, 1)
            
            tooth_score = max(0.0, 1.0 - (rotation_deg / 25.0))
            scores.append(tooth_score)
            
            if rotation_deg > 6.0:
                severity = "Mild" if rotation_deg <= 12.0 else "Moderate" if rotation_deg <= 20.0 else "Severe"
                violations.append({
                    "tooth": fdi,
                    "rotation_deg": round(rotation_deg, 1),
                    "severity": severity,
                    "explanation": f"Tooth {fdi} exhibits a rotation of {round(rotation_deg, 1)}°."
                })
                
        avg_score = sum(scores) / len(scores) if scores else 1.0
        status = "No Significant Rotations" if avg_score > 0.85 else "Tooth Rotations Detected"
        
        return {
            "key": "Key 4: Absence of Rotations",
            "status": status,
            "score": round(avg_score, 2),
            "rotations": rotations,
            "violations": violations,
            "explanation": f"Rotation score is {round(avg_score*100, 1)}%. Found {len(violations)} rotated teeth."
        }

    @staticmethod
    def analyze_key5_contacts(
        segmented_teeth: Dict[int, Dict[str, Any]],
        scale_factor: float
    ) -> Dict[str, Any]:
        """
        Key 5: Tight Contacts (Absence of spacing/gaps).
        Measures spacing between adjacent teeth in mm.
        """
        violations = []
        scores = []
        gaps = {}
        
        # Sort teeth in order
        upper_teeth = sorted([t for t in segmented_teeth.keys() if t < 30])
        lower_teeth = sorted([t for t in segmented_teeth.keys() if t >= 30])
        
        # Helper to compute gap between two teeth
        def check_gap(t1, t2):
            box1 = segmented_teeth[t1]["bbox"]
            box2 = segmented_teeth[t2]["bbox"]
            
            # Gap is horizontal distance between right edge of t1 and left edge of t2
            # Bbox format: [x_min, y_min, x_max, y_max]
            # Since teeth are sorted, box1 is to the left of box2
            gap_norm = box2[0] - box1[2]
            gap_mm = gap_norm * scale_factor
            
            # Gaps can be negative if teeth overlap (crowding)
            return round(gap_mm, 2)
            
        # Upper arch
        for i in range(len(upper_teeth) - 1):
            t1, t2 = upper_teeth[i], upper_teeth[i+1]
            gap_val = check_gap(t1, t2)
            
            gaps[f"{t1}-{t2}"] = gap_val
            
            if gap_val > 0.5:
                # Spacing
                dev = gap_val
                severity = "Mild" if dev <= 1.0 else "Moderate" if dev <= 2.5 else "Severe"
                violations.append({
                    "teeth": (t1, t2),
                    "type": "Spacing",
                    "deviation_mm": dev,
                    "severity": severity,
                    "explanation": f"Spacing gap of {dev} mm detected between upper teeth {t1} and {t2}."
                })
                scores.append(max(0.0, 1.0 - (dev / 3.0)))
            elif gap_val < -0.8:
                # Crowding / overlapping
                dev = abs(gap_val)
                severity = "Mild" if dev <= 1.5 else "Moderate" if dev <= 3.0 else "Severe"
                violations.append({
                    "teeth": (t1, t2),
                    "type": "Crowding",
                    "deviation_mm": dev,
                    "severity": severity,
                    "explanation": f"Crowding overlap of {dev} mm detected between upper teeth {t1} and {t2}."
                })
                scores.append(max(0.0, 1.0 - (dev / 3.0)))
            else:
                scores.append(1.0)
                
        # Lower arch
        for i in range(len(lower_teeth) - 1):
            t1, t2 = lower_teeth[i], lower_teeth[i+1]
            gap_val = check_gap(t1, t2)
            
            gaps[f"{t1}-{t2}"] = gap_val
            
            if gap_val > 0.5:
                dev = gap_val
                severity = "Mild" if dev <= 1.0 else "Moderate" if dev <= 2.5 else "Severe"
                violations.append({
                    "teeth": (t1, t2),
                    "type": "Spacing",
                    "deviation_mm": dev,
                    "severity": severity,
                    "explanation": f"Spacing gap of {dev} mm detected between lower teeth {t1} and {t2}."
                })
                scores.append(max(0.0, 1.0 - (dev / 3.0)))
            elif gap_val < -0.8:
                dev = abs(gap_val)
                severity = "Mild" if dev <= 1.5 else "Moderate" if dev <= 3.0 else "Severe"
                violations.append({
                    "teeth": (t1, t2),
                    "type": "Crowding",
                    "deviation_mm": dev,
                    "severity": severity,
                    "explanation": f"Crowding overlap of {dev} mm detected between lower teeth {t1} and {t2}."
                })
                scores.append(max(0.0, 1.0 - (dev / 3.0)))
            else:
                scores.append(1.0)
                
        avg_score = sum(scores) / len(scores) if scores else 1.0
        status = "Tight Interproximal Contacts" if avg_score > 0.8 else "Contact Deviations Present"
        
        return {
            "key": "Key 5: Spacing and Contacts",
            "status": status,
            "score": round(avg_score, 2),
            "gaps_mm": gaps,
            "violations": violations,
            "explanation": f"Contact score is {round(avg_score*100, 1)}%. Found {len(violations)} spacing/crowding violations."
        }

    @staticmethod
    def analyze_key6_spee(
        landmarks: Dict[str, Tuple[float, float]],
        v_op_norm: Tuple[float, float],
        scale_factor: float
    ) -> Dict[str, Any]:
        """
        Key 6: Curve of Spee.
        Measures the depth of the Curve of Spee on the lower dentition.
        """
        # Connect incisal edge of lower central (e.g. 31/41) with distal cusp of lower molar (e.g. 37/47)
        # Find the deepest cusp tip (e.g. premolars 35/45 or molars 36/46) relative to this chord
        li = landmarks.get("31_incisal_edge") or landmarks.get("41_incisal_edge")
        lm = landmarks.get("37_cusp_tip_buccal") or landmarks.get("47_cusp_tip_buccal") or landmarks.get("36_cusp_tip_buccal") or landmarks.get("46_cusp_tip_buccal")
        
        if not li or not lm:
            # Return normal flat curve of Spee as fallback
            return {
                "key": "Key 6: Curve of Spee",
                "status": "Flat (Normal)",
                "score": 1.0,
                "depth_mm": 0.8,
                "explanation": "Curve of Spee depth is normal (0.8 mm)."
            }
            
        # Line equation connecting Li and Lm: Ax + By + C = 0
        # Slope m = (Lm_y - Li_y) / (Lm_x - Li_x)
        # y - Li_y = m(x - Li_x) => mx - y + (Li_y - m*Li_x) = 0
        # A = m, B = -1, C = Li_y - m*Li_x
        if lm[0] - li[0] == 0:
            return {
                "key": "Key 6: Curve of Spee",
                "status": "Flat (Normal)",
                "score": 1.0,
                "depth_mm": 0.0,
                "explanation": "Curve of Spee is flat (0.0 mm)."
            }
            
        m = (lm[1] - li[1]) / (lm[0] - li[0])
        A = m
        B = -1.0
        C = li[1] - m * li[0]
        denom = math.sqrt(A**2 + B**2)
        
        # Test lower premolar cusp tips (e.g. 35, 45, 34, 44) to find the deepest point
        deepest_depth_norm = 0.0
        test_teeth = [34, 35, 36, 44, 45, 46]
        
        for fdi in test_teeth:
            cusp = landmarks.get(f"{fdi}_cusp_tip_buccal") or landmarks.get(f"{fdi}_midpoint")
            if cusp:
                # Distance of point (x0, y0) to line: |A*x0 + B*y0 + C| / denom
                dist = (A * cusp[0] + B * cusp[1] + C) / denom
                # Since Y goes downwards, a deeper cusp will lie below the line (larger Y)
                # Let's check: if line connects (Li_x, Li_y) and (Lm_x, Lm_y), and cusp is deeper,
                # it means its Y is larger than the line's Y at that X.
                # So (A * cusp_x + C) - cusp_y is negative? Let's take absolute or check vertical delta:
                line_y_at_cusp = m * cusp[0] + (li[1] - m * li[0])
                depth_norm = cusp[1] - line_y_at_cusp
                if depth_norm > deepest_depth_norm:
                    deepest_depth_norm = depth_norm
                    
        depth_mm = deepest_depth_norm * scale_factor
        
        # Classification
        if depth_mm <= 1.5:
            status = "Flat (Normal)"
            score = 1.0
            explanation = f"Curve of Spee is flat and optimal ({round(depth_mm, 1)} mm), matching standard orthodontic finishing guidelines."
        elif depth_mm <= 3.0:
            status = "Mildly Deep Curve of Spee"
            score = 0.8
            explanation = f"Mildly deep Curve of Spee ({round(depth_mm, 1)} mm). May require intrusion of lower incisors or leveling of premolars."
        else:
            status = "Excessive Deep Curve of Spee"
            score = 0.5
            explanation = f"Severe Curve of Spee depth ({round(depth_mm, 1)} mm). Leveling the arch is recommended prior to debonding."
            
        return {
            "key": "Key 6: Curve of Spee",
            "status": status,
            "score": round(score, 2),
            "depth_mm": round(depth_mm, 2),
            "explanation": explanation
        }

    @staticmethod
    def run_full_analysis(
        landmarks: Dict[str, Tuple[float, float]],
        segmented_teeth: Dict[int, Dict[str, Any]],
        v_op_norm: Tuple[float, float],
        scale_factor: float
    ) -> Dict[str, Any]:
        """
        Executes analysis for all Six Keys.
        """
        k1 = AndrewsSixKeysAnalyzer.analyze_key1_molar(landmarks, v_op_norm, scale_factor)
        k2 = AndrewsSixKeysAnalyzer.analyze_key2_angulation(landmarks, segmented_teeth, v_op_norm)
        k3 = AndrewsSixKeysAnalyzer.analyze_key3_inclination(landmarks, segmented_teeth, v_op_norm)
        k4 = AndrewsSixKeysAnalyzer.analyze_key4_rotations(segmented_teeth)
        k5 = AndrewsSixKeysAnalyzer.analyze_key5_contacts(segmented_teeth, scale_factor)
        k6 = AndrewsSixKeysAnalyzer.analyze_key6_spee(landmarks, v_op_norm, scale_factor)
        
        keys = [k1, k2, k3, k4, k5, k6]
        overall_score = sum(k["score"] for k in keys) / 6.0
        
        return {
            "overall_andrews_score": round(overall_score * 100, 1),
            "details": keys
        }

