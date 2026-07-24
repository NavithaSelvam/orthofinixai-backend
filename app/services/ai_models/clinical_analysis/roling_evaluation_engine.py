from typing import Dict, Any

class RolingEvaluationEngine:
    """
    Evaluates Rebecca Roling's criteria using ClinicalMeasurementEngine outputs.
    Provides detailed categorical evaluations.
    """
    
    def evaluate(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "Functional finishing": self._evaluate_functional(measurements),
            "Arch harmony": self._evaluate_arch_harmony(measurements),
            "Torque optimization": self._evaluate_torque(measurements),
            "Occlusal stability": self._evaluate_stability(measurements),
            "Second molar finishing": self._evaluate_second_molars(measurements)
        }

    def _evaluate_functional(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        oj = measurements.get("overjet_mm", 0.0)
        ob = measurements.get("overbite_mm", 0.0)
        contacts = measurements.get("occlusal_contacts", {})
        
        issues = []
        if oj > 1.0 or oj < 0.0: issues.append(f"Improper overjet ({oj}mm)")
        if ob < 0.0 or ob > 3.0: issues.append(f"Improper overbite ({ob}mm)")
        
        missed_contacts = [pair for pair, contact in contacts.items() if not contact]
        if missed_contacts:
            issues.append(f"Missing functional contacts on {len(missed_contacts)} pairs")
            
        if not issues:
            return {"status": "Excellent", "details": "Ideal overjet, overbite, and posterior functional contacts."}
        return {"status": "Needs Improvement", "details": "; ".join(issues)}

    def _evaluate_arch_harmony(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        sym = measurements.get("arch_symmetry_index", 1.0)
        midline = measurements.get("midline_deviation_mm", 0.0)
        
        issues = []
        if sym < 0.9: issues.append(f"Arch asymmetry detected (index {sym})")
        if midline > 0.5: issues.append(f"Midline deviation of {midline}mm")
        
        if not issues:
            return {"status": "Excellent", "details": "Arch is harmonious and symmetric with coincident midlines."}
        return {"status": "Needs Improvement", "details": "; ".join(issues)}

    def _evaluate_torque(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        inclinations = measurements.get("buccolingual_inclination_degrees", {})
        affected = [t for t, a in inclinations.items() if abs(a) > 5.0]
        
        if affected:
            return {"status": "Needs Improvement", "details": f"Torque optimization needed on teeth: {', '.join(affected)}"}
        return {"status": "Excellent", "details": "Torque is optimized across the arch."}

    def _evaluate_stability(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        ridges = measurements.get("marginal_ridge_discrepancies_mm", {})
        affected = [pair for pair, diff in ridges.items() if diff > 0.5]
        
        if affected:
            return {"status": "Needs Improvement", "details": f"Marginal ridge steps affecting stability: {', '.join(affected)}"}
        return {"status": "Excellent", "details": "Marginal ridges are level, supporting occlusal stability."}

    def _evaluate_second_molars(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        angulations = measurements.get("root_angulation_degrees", {})
        ridges = measurements.get("marginal_ridge_discrepancies_mm", {})
        
        issues = []
        molar_angulations = {t: a for t, a in angulations.items() if t in ["17", "27", "37", "47"]}
        for t, a in molar_angulations.items():
            if abs(a) > 5.0: issues.append(f"Tooth {t} angulation ({a}deg)")
            
        molar_ridges = {pair: diff for pair, diff in ridges.items() if "7" in pair}
        for pair, diff in molar_ridges.items():
            if diff > 0.5: issues.append(f"Ridge discrepancy {pair} ({diff}mm)")
            
        if issues:
            return {"status": "Needs Improvement", "details": "; ".join(issues)}
        return {"status": "Excellent", "details": "Second molars are well finished and leveled."}
