from typing import Dict, Any

class RaleighWilliamsEngine:
    """
    Evaluates Raleigh Williams criteria using clinical measurements.
    """
    
    def evaluate(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        result = {
            "categories": {},
            "score": 100
        }
        
        # 1. Axial inclination
        inc = self._evaluate_axial_inclination(measurements)
        result["categories"]["Axial inclination"] = inc
        result["score"] -= inc["deduction"]
        
        # 2. Interarch relationships
        rel = self._evaluate_interarch_relationships(measurements)
        result["categories"]["Interarch relationships"] = rel
        result["score"] -= rel["deduction"]
        
        # 3. Curve of Spee
        spee = self._evaluate_curve_of_spee(measurements)
        result["categories"]["Curve of Spee"] = spee
        result["score"] -= spee["deduction"]
        
        # 4. Midline
        mid = self._evaluate_midline(measurements)
        result["categories"]["Midline"] = mid
        result["score"] -= mid["deduction"]
        
        # 5. Contacts
        contacts = self._evaluate_contacts(measurements)
        result["categories"]["Contacts"] = contacts
        result["score"] -= contacts["deduction"]
        
        # 6. Arch Form
        arch = self._evaluate_arch_form(measurements)
        result["categories"]["Arch Form"] = arch
        result["score"] -= arch["deduction"]
        
        result["score"] = max(0, result["score"])
        return result

    def _evaluate_axial_inclination(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        angulations = measurements.get("root_angulation_degrees", {})
        inclinations = measurements.get("buccolingual_inclination_degrees", {})
        
        deduction = 0
        issues = []
        for t, ang in angulations.items():
            if abs(ang) > 5.0:
                deduction += 2
                issues.append(f"T{t} mesiodistal angulation ({ang}deg)")
                
        for t, inc in inclinations.items():
            if abs(inc) > 5.0:
                deduction += 2
                issues.append(f"T{t} buccolingual inclination ({inc}deg)")
                
        if not issues:
            return {"comments": "Ideal axial inclinations.", "deduction": 0}
        return {"comments": "; ".join(issues), "deduction": min(deduction, 20)}

    def _evaluate_interarch_relationships(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        oj = measurements.get("overjet_mm", 0.0)
        ob = measurements.get("overbite_mm", 0.0)
        rels = measurements.get("molar_relationship_mm", {})
        
        deduction = 0
        issues = []
        
        if oj < 0.0 or oj > 2.0:
            deduction += 5
            issues.append(f"Improper overjet ({oj}mm)")
        if ob < 0.0 or ob > 3.0:
            deduction += 5
            issues.append(f"Improper overbite ({ob}mm)")
            
        for side, dist in rels.items():
            if abs(dist) > 2.0:
                deduction += 5
                issues.append(f"Class deviation on {side} molar ({dist}mm)")
                
        if not issues:
            return {"comments": "Ideal interarch relationships.", "deduction": 0}
        return {"comments": "; ".join(issues), "deduction": min(deduction, 25)}

    def _evaluate_curve_of_spee(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        depth = measurements.get("curve_of_spee_depth_mm", 0.0)
        
        if depth > 1.5:
            return {"comments": f"Curve of Spee is excessively deep ({depth}mm)", "deduction": 10}
        return {"comments": f"Curve of Spee is within normal limits ({depth}mm)", "deduction": 0}

    def _evaluate_midline(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        dev = measurements.get("midline_deviation_mm", 0.0)
        if dev > 1.0:
            return {"comments": f"Midline deviation detected ({dev}mm)", "deduction": 10}
        return {"comments": "Midlines are coincident.", "deduction": 0}

    def _evaluate_contacts(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        gaps = measurements.get("interproximal_contacts_mm", {})
        occ = measurements.get("occlusal_contacts", {})
        
        deduction = 0
        issues = []
        
        for pair, gap in gaps.items():
            if gap > 0.5:
                deduction += 3
                issues.append(f"Open contact at {pair}")
                
        for pair, contact in occ.items():
            if not contact:
                deduction += 2
                issues.append(f"Missing occlusal contact at {pair}")
                
        if not issues:
            return {"comments": "Tight interproximal and occlusal contacts.", "deduction": 0}
        return {"comments": "; ".join(issues), "deduction": min(deduction, 20)}

    def _evaluate_arch_form(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        sym = measurements.get("arch_symmetry_index", 1.0)
        if sym < 0.9:
            return {"comments": f"Asymmetric arch form (index {sym})", "deduction": 15}
        return {"comments": "Symmetrical arch form.", "deduction": 0}
