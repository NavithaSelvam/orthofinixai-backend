from typing import Dict, Any

class AndrewsEvaluationEngine:
    """
    Evaluates Andrews Six Keys using ClinicalMeasurementEngine outputs.
    Returns structured Pass/Fail status without clinical recommendations.
    """
    
    def evaluate(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "Key 1": self._evaluate_key1(measurements),
            "Key 2": self._evaluate_key2(measurements),
            "Key 3": self._evaluate_key3(measurements),
            "Key 4": self._evaluate_key4(measurements),
            "Key 5": self._evaluate_key5(measurements),
            "Key 6": self._evaluate_key6(measurements)
        }

    def _evaluate_key1(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        # Key 1: Molar Relationship (within 1.5mm is Pass)
        rels = measurements.get("molar_relationship_mm", {})
        affected = []
        fail_reasons = []
        
        for side, dist in rels.items():
            if abs(dist) > 1.5:
                fail_reasons.append(f"{side} side deviated by {dist}mm")
                if side == "right": affected.extend(["16", "46"])
                if side == "left": affected.extend(["26", "36"])
                
        if fail_reasons:
            return {"status": "Fail", "reason": "; ".join(fail_reasons), "affected_teeth": affected}
        return {"status": "Pass", "reason": "Molar relationship is ideal", "affected_teeth": []}

    def _evaluate_key2(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        # Key 2: Crown Angulation
        angulations = measurements.get("root_angulation_degrees", {})
        affected = []
        for tooth, angle in angulations.items():
            if abs(angle) > 5.0: # simplistic generic threshold
                affected.append(tooth)
                
        if affected:
            return {"status": "Fail", "reason": f"{len(affected)} teeth exceed angulation limits", "affected_teeth": affected}
        return {"status": "Pass", "reason": "Crown angulations are acceptable", "affected_teeth": []}

    def _evaluate_key3(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        # Key 3: Crown Inclination
        inclinations = measurements.get("buccolingual_inclination_degrees", {})
        affected = []
        for tooth, angle in inclinations.items():
            if abs(angle) > 5.0:
                affected.append(tooth)
                
        if affected:
            return {"status": "Fail", "reason": f"{len(affected)} teeth exceed inclination limits", "affected_teeth": affected}
        return {"status": "Pass", "reason": "Crown inclinations are acceptable", "affected_teeth": []}

    def _evaluate_key4(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        # Key 4: Rotations
        rotations = measurements.get("rotation_degrees", {})
        affected = []
        for tooth, angle in rotations.items():
            if abs(angle) > 5.0:
                affected.append(tooth)
                
        if affected:
            return {"status": "Fail", "reason": f"{len(affected)} teeth exhibit significant rotation", "affected_teeth": affected}
        return {"status": "Pass", "reason": "No significant rotations detected", "affected_teeth": []}

    def _evaluate_key5(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        # Key 5: Tight Contacts
        gaps = measurements.get("interproximal_contacts_mm", {})
        affected = []
        for pair, gap in gaps.items():
            if gap > 0.5:
                for t in pair.split("-"):
                    if t not in affected: affected.append(t)
                    
        if affected:
            return {"status": "Fail", "reason": "Open contacts or spacing detected", "affected_teeth": affected}
        return {"status": "Pass", "reason": "Contacts are tight", "affected_teeth": []}

    def _evaluate_key6(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        # Key 6: Curve of Spee
        depth = measurements.get("curve_of_spee_depth_mm", 0.0)
        # Ideal is flat to 1.5mm
        if depth > 1.5:
            return {"status": "Fail", "reason": f"Curve of Spee depth is {depth}mm (too deep)", "affected_teeth": []}
        return {"status": "Pass", "reason": f"Curve of Spee depth is {depth}mm (acceptable)", "affected_teeth": []}
