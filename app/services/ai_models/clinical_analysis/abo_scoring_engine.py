from typing import Dict, Any, List

class AboScoringEngine:
    """
    Evaluates the measurements provided by the ClinicalMeasurementEngine 
    against ABO Objective Grading System criteria and computes deductions.
    Highlights affected teeth.
    """
    
    def score(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        result = {
            "total_score": 0,
            "categories": {}
        }
        
        # 1. Alignment
        alignment = self._score_alignment(measurements)
        result["categories"]["alignment"] = alignment
        result["total_score"] += alignment["penalty"]
        
        # 2. Marginal Ridge
        ridge = self._score_marginal_ridge(measurements)
        result["categories"]["marginal_ridge"] = ridge
        result["total_score"] += ridge["penalty"]
        
        # 3. Buccolingual Inclination
        inclination = self._score_buccolingual_inclination(measurements)
        result["categories"]["buccolingual_inclination"] = inclination
        result["total_score"] += inclination["penalty"]
        
        # 4. Occlusal Contacts
        occlusal = self._score_occlusal_contacts(measurements)
        result["categories"]["occlusal_contacts"] = occlusal
        result["total_score"] += occlusal["penalty"]
        
        # 5. Interproximal Contacts
        interproximal = self._score_interproximal_contacts(measurements)
        result["categories"]["interproximal_contacts"] = interproximal
        result["total_score"] += interproximal["penalty"]
        
        # 6. Overjet
        overjet = self._score_overjet(measurements)
        result["categories"]["overjet"] = overjet
        result["total_score"] += overjet["penalty"]
        
        # 7. Root Angulation
        root_angulation = self._score_root_angulation(measurements)
        result["categories"]["root_angulation"] = root_angulation
        result["total_score"] += root_angulation["penalty"]
        
        return result

    def _score_alignment(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        rotations = measurements.get("rotation_degrees", {})
        penalty = 0
        affected = []
        for tooth, angle in rotations.items():
            if abs(angle) > 5.0: # threshold for rotation penalty
                penalty += 1 # standard 1 point deduction
                affected.append(tooth)
                
        crowding = measurements.get("crowding_mm", 0.0)
        if crowding > 1.0:
            penalty += int(crowding) 
            
        return {"penalty": penalty, "affected_teeth": affected}

    def _score_marginal_ridge(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        discrepancies = measurements.get("marginal_ridge_discrepancies_mm", {})
        penalty = 0
        affected = []
        for pair, diff in discrepancies.items():
            if diff > 0.5: # ABO allows 0.5mm discrepancy usually
                penalty += 1
                teeth = pair.split("-")
                for t in teeth:
                    if t not in affected:
                        affected.append(t)
                        
        return {"penalty": penalty, "affected_teeth": affected}

    def _score_buccolingual_inclination(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        inclinations = measurements.get("buccolingual_inclination_degrees", {})
        penalty = 0
        affected = []
        for tooth, angle in inclinations.items():
            if abs(angle) > 5.0:
                penalty += 1
                affected.append(tooth)
                
        return {"penalty": penalty, "affected_teeth": affected}

    def _score_occlusal_contacts(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        contacts = measurements.get("occlusal_contacts", {})
        penalty = 0
        affected = []
        for pair, in_contact in contacts.items():
            if not in_contact:
                penalty += 1
                teeth = pair.split("-")
                for t in teeth:
                    if t not in affected:
                        affected.append(t)
                        
        return {"penalty": penalty, "affected_teeth": affected}

    def _score_interproximal_contacts(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        gaps = measurements.get("interproximal_contacts_mm", {})
        penalty = 0
        affected = []
        for pair, gap in gaps.items():
            if gap > 0.5: # > 0.5mm gap means contact is open
                penalty += 1
                teeth = pair.split("-")
                for t in teeth:
                    if t not in affected:
                        affected.append(t)
                        
        return {"penalty": penalty, "affected_teeth": affected}

    def _score_overjet(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        oj = measurements.get("overjet_mm", 0.0)
        penalty = 0
        affected = []
        if oj < 0.0 or oj > 1.0:
            penalty += 1
            affected = ["11", "21", "31", "41"]
            
        return {"penalty": penalty, "affected_teeth": affected}

    def _score_root_angulation(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        angulations = measurements.get("root_angulation_degrees", {})
        penalty = 0
        affected = []
        for tooth, angle in angulations.items():
            if abs(angle) > 5.0:
                penalty += 1
                affected.append(tooth)
                
        return {"penalty": penalty, "affected_teeth": affected}
