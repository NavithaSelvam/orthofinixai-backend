import math
from typing import Dict, Tuple, Any, List
from .geometry import calculate_distance, calculate_angle_between_vectors, project_vector_magnitude, fit_occlusal_plane

class ClinicalMeasurementEngine:
    """
    Computes various orthodontic clinical measurements using 2D landmarks.
    Provides structured numeric outputs without generating clinical recommendations.
    """

    def __init__(self, scale_factor: float = 1.0):
        """
        Initializes the engine with a scale factor.
        :param scale_factor: A multiplier to convert normalized or pixel distances into millimeters.
        """
        self.scale_factor = scale_factor

    def analyze(self, landmarks: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """
        Runs all calculations and returns a dictionary of structured measurements.
        """
        # Determine occlusal plane (if sufficient points are available)
        op_points = []
        for key, pt in landmarks.items():
            if "incisal_edge" in key or "cusp_tip" in key:
                op_points.append(pt)
        
        _, v_op_norm = fit_occlusal_plane(op_points)

        results = {
            "overjet_mm": self._calculate_overjet(landmarks, v_op_norm),
            "overbite_mm": self._calculate_overbite(landmarks, v_op_norm),
            "midline_deviation_mm": self._calculate_midline(landmarks),
            "rotation_degrees": self._calculate_rotation(landmarks),
            "crowding_mm": self._calculate_crowding(landmarks),
            "spacing_mm": self._calculate_spacing(landmarks),
            "curve_of_spee_depth_mm": self._calculate_curve_of_spee(landmarks),
            "marginal_ridge_discrepancies_mm": self._calculate_marginal_ridge(landmarks),
            "arch_width_mm": self._calculate_arch_width(landmarks),
            "arch_symmetry_index": self._calculate_arch_symmetry(landmarks),
            "buccolingual_inclination_degrees": self._calculate_buccolingual_inclination(landmarks, v_op_norm),
            "root_angulation_degrees": self._calculate_root_angulation(landmarks, v_op_norm),
            "interproximal_contacts_mm": self._calculate_interproximal_contacts(landmarks),
            "occlusal_contacts": self._calculate_occlusal_contacts(landmarks),
            "molar_relationship_mm": self._calculate_molar_relationship(landmarks, v_op_norm)
        }
        return results

    def _calculate_overjet(self, landmarks: Dict[str, Tuple[float, float]], v_op_norm: Tuple[float, float]) -> float:
        ui = landmarks.get("11_incisal_edge") or landmarks.get("21_incisal_edge")
        li = landmarks.get("41_incisal_edge") or landmarks.get("31_incisal_edge")
        if not ui or not li:
            return 0.0
        
        ll_offset_x = -0.015 
        ll = (li[0] + ll_offset_x, li[1])
        
        v_oj = (ui[0] - ll[0], ui[1] - ll[1])
        oj_normalized = project_vector_magnitude(v_oj, v_op_norm)
        return round(oj_normalized * self.scale_factor, 2)

    def _calculate_overbite(self, landmarks: Dict[str, Tuple[float, float]], v_op_norm: Tuple[float, float]) -> float:
        ui = landmarks.get("11_incisal_edge") or landmarks.get("21_incisal_edge")
        li = landmarks.get("41_incisal_edge") or landmarks.get("31_incisal_edge")
        if not ui or not li:
            return 0.0
            
        ux, uy = v_op_norm
        n_op_norm = (-uy, ux) # upward normal

        v_ob = (ui[0] - li[0], ui[1] - li[1])
        ob_normalized = project_vector_magnitude(v_ob, n_op_norm)
        
        ob_mm = abs(ob_normalized * self.scale_factor)
        if ui[1] > li[1]: # open bite
            ob_mm = -ob_mm
        return round(ob_mm, 2)

    def _calculate_midline(self, landmarks: Dict[str, Tuple[float, float]]) -> float:
        u11 = landmarks.get("11_cej_mesial") or landmarks.get("11_incisal_edge")
        u21 = landmarks.get("21_cej_mesial") or landmarks.get("21_incisal_edge")
        l41 = landmarks.get("41_cej_mesial") or landmarks.get("41_incisal_edge")
        l31 = landmarks.get("31_cej_mesial") or landmarks.get("31_incisal_edge")

        if not u11 or not u21 or not l41 or not l31:
            return 0.0

        u_mid_x = (u11[0] + u21[0]) / 2.0
        l_mid_x = (l41[0] + l31[0]) / 2.0
        
        diff_x = abs(u_mid_x - l_mid_x)
        return round(diff_x * self.scale_factor, 2)

    def _calculate_rotation(self, landmarks: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        rotations = {}
        for fdi in range(11, 49):
            mesial = landmarks.get(f"{fdi}_cej_mesial")
            distal = landmarks.get(f"{fdi}_cej_distal")
            if mesial and distal:
                dx = distal[0] - mesial[0]
                dy = distal[1] - mesial[1]
                angle = math.degrees(math.atan2(dy, dx + 1e-9))
                rotations[str(fdi)] = round(angle, 2)
        return rotations

    def _calculate_crowding(self, landmarks: Dict[str, Tuple[float, float]]) -> float:
        # Placeholder for crowding metric based on 2D projections.
        # A true 3D arch calculation is required for clinical crowding.
        return 0.0

    def _calculate_spacing(self, landmarks: Dict[str, Tuple[float, float]]) -> float:
        # Placeholder for spacing metric. 
        # A true 3D arch calculation is required for clinical spacing.
        return 0.0

    def _calculate_curve_of_spee(self, landmarks: Dict[str, Tuple[float, float]]) -> float:
        l1 = landmarks.get("41_incisal_edge")
        l7 = landmarks.get("47_cusp_tip_buccal") or landmarks.get("46_cusp_tip_buccal")
        
        if not l1 or not l7:
            return 0.0
            
        m = (l7[1] - l1[1]) / (l7[0] - l1[0] + 1e-6)
        c = l1[1] - m * l1[0]
        denom = math.sqrt(m**2 + 1)
        
        max_depth = 0.0
        for fdi in [43, 44, 45, 46]:
            pt = landmarks.get(f"{fdi}_cusp_tip_buccal") or landmarks.get(f"{fdi}_cusp_tip")
            if pt:
                dist = abs(m * pt[0] - pt[1] + c) / denom
                max_depth = max(max_depth, dist)
                
        return round(max_depth * self.scale_factor, 2)

    def _calculate_marginal_ridge(self, landmarks: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        discrepancies = {}
        # Compare adjacent posterior teeth (e.g. 45-46, 46-47, etc.)
        for fdi_base in [14, 15, 16, 24, 25, 26, 34, 35, 36, 44, 45, 46]:
            m1 = landmarks.get(f"{fdi_base}_fa") or landmarks.get(f"{fdi_base}_midpoint")
            m2 = landmarks.get(f"{fdi_base+1}_fa") or landmarks.get(f"{fdi_base+1}_midpoint")
            if m1 and m2:
                diff = abs(m1[1] - m2[1]) * self.scale_factor
                discrepancies[f"{fdi_base}-{fdi_base+1}"] = round(diff, 2)
        return discrepancies

    def _calculate_arch_width(self, landmarks: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        widths = {}
        # Intercanine
        u13 = landmarks.get("13_cusp_tip")
        u23 = landmarks.get("23_cusp_tip")
        if u13 and u23:
            widths["upper_intercanine"] = round(calculate_distance(u13, u23) * self.scale_factor, 2)
            
        # Intermolar
        u16 = landmarks.get("16_cusp_tip_buccal")
        u26 = landmarks.get("26_cusp_tip_buccal")
        if u16 and u26:
            widths["upper_intermolar"] = round(calculate_distance(u16, u26) * self.scale_factor, 2)
            
        return widths

    def _calculate_arch_symmetry(self, landmarks: Dict[str, Tuple[float, float]]) -> float:
        u11 = landmarks.get("11_incisal_edge")
        u21 = landmarks.get("21_incisal_edge")
        u16 = landmarks.get("16_cusp_tip_buccal")
        u26 = landmarks.get("26_cusp_tip_buccal")
        
        if u11 and u21 and u16 and u26:
            mid_x = (u11[0] + u21[0]) / 2.0
            mid_y = (u11[1] + u21[1]) / 2.0
            midline = (mid_x, mid_y)
            
            d_right = calculate_distance(midline, u16)
            d_left = calculate_distance(midline, u26)
            
            if d_left > 0:
                symmetry_index = min(d_right, d_left) / max(d_right, d_left)
                return round(symmetry_index, 2)
        return 1.0

    def _calculate_buccolingual_inclination(self, landmarks: Dict[str, Tuple[float, float]], v_op_norm: Tuple[float, float]) -> Dict[str, float]:
        inclinations = {}
        ux, uy = v_op_norm
        n_op_norm = (-uy, ux) 
        
        for fdi in range(11, 49):
            fa = landmarks.get(f"{fdi}_fa")
            apex = landmarks.get(f"{fdi}_apex")
            if fa and apex:
                v_tooth = (fa[0] - apex[0], fa[1] - apex[1])
                angle = calculate_angle_between_vectors(v_tooth, n_op_norm)
                inclinations[str(fdi)] = round(angle, 2)
                
        return inclinations

    def _calculate_root_angulation(self, landmarks: Dict[str, Tuple[float, float]], v_op_norm: Tuple[float, float]) -> Dict[str, float]:
        angulations = {}
        # Angulation in mesiodistal plane (frontal view typically).
        # We can approximate by measuring angle of fa->apex relative to the vertical normal.
        ux, uy = v_op_norm
        n_op_norm = (-uy, ux) 
        
        for fdi in range(11, 49):
            fa = landmarks.get(f"{fdi}_fa")
            apex = landmarks.get(f"{fdi}_apex")
            if fa and apex:
                # Same underlying vector as buccolingual but in a 2D projection, 
                # we just use the raw angle off the vertical.
                v_tooth = (fa[0] - apex[0], fa[1] - apex[1])
                angle = calculate_angle_between_vectors(v_tooth, n_op_norm)
                angulations[str(fdi)] = round(angle, 2)
                
        return angulations

    def _calculate_interproximal_contacts(self, landmarks: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        contacts = {}
        # Adjacent teeth gaps
        for fdi_base in range(11, 48):
            if fdi_base % 10 == 8: # skip 8 to next quadrant
                continue
            distal = landmarks.get(f"{fdi_base}_cej_distal")
            mesial = landmarks.get(f"{fdi_base+1}_cej_mesial")
            if distal and mesial:
                gap = calculate_distance(distal, mesial) * self.scale_factor
                contacts[f"{fdi_base}-{fdi_base+1}"] = round(gap, 2)
        return contacts

    def _calculate_occlusal_contacts(self, landmarks: Dict[str, Tuple[float, float]]) -> Dict[str, bool]:
        contacts = {}
        # Check if upper cusp tip is close to lower groove or opposing cusp
        # Simplification for 2D: distance in Y axis
        pairs = [(16, 46), (15, 45), (14, 44), (26, 36), (25, 35), (24, 34)]
        for u, l in pairs:
            u_pt = landmarks.get(f"{u}_cusp_tip_buccal") or landmarks.get(f"{u}_cusp_tip")
            l_pt = landmarks.get(f"{l}_cusp_tip_buccal") or landmarks.get(f"{l}_cusp_tip")
            if u_pt and l_pt:
                y_dist = abs(u_pt[1] - l_pt[1]) * self.scale_factor
                contacts[f"{u}-{l}"] = bool(y_dist < 1.0) # Within 1mm is in contact
        return contacts

    def _calculate_molar_relationship(self, landmarks: Dict[str, Tuple[float, float]], v_op_norm: Tuple[float, float]) -> Dict[str, float]:
        rels = {}
        # Right Side (16 vs 46)
        u16 = landmarks.get("16_cusp_tip_buccal") or landmarks.get("16_midpoint")
        l46 = landmarks.get("46_buccal_groove") or landmarks.get("46_midpoint")
        if u16 and l46:
            dx = u16[0] - l46[0]
            dy = u16[1] - l46[1]
            dist_norm = dx * v_op_norm[0] + dy * v_op_norm[1]
            rels["right"] = round(-dist_norm * self.scale_factor, 2) # Adjust sign for mesial/distal
            
        # Left Side (26 vs 36)
        u26 = landmarks.get("26_cusp_tip_buccal") or landmarks.get("26_midpoint")
        l36 = landmarks.get("36_buccal_groove") or landmarks.get("36_midpoint")
        if u26 and l36:
            dx = u26[0] - l36[0]
            dy = u26[1] - l36[1]
            dist_norm = dx * v_op_norm[0] + dy * v_op_norm[1]
            rels["left"] = round(dist_norm * self.scale_factor, 2) # Left side has opposite mesial direction
            
        return rels
