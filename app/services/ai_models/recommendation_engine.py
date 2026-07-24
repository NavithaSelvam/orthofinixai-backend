from typing import List, Dict, Any

class ClinicalRecommendationEngine:
    """
    Dynamic rule-based recommendation engine that generates personalized 
    treatment mechanics based strictly on precise clinical measurements.
    """
    
    @staticmethod
    def generate_recommendations(measurements: Dict[str, Any]) -> List[str]:
        recommendations = []
        
        # 1. Open contacts
        gaps = measurements.get("interproximal_contacts_mm", {})
        for pair, gap in gaps.items():
            if gap > 0.5:
                recommendations.append(f"Open contact detected at {pair} ({gap}mm) -> Apply elastomeric chain to close space.")
                
        # 2. Rotations
        rotations = measurements.get("rotation_degrees", {})
        for tooth, angle in rotations.items():
            if abs(angle) > 5.0:
                recommendations.append(f"Rotation on tooth {tooth} ({angle}°) -> Bracket repositioning required.")
                
        # 3. High Overjet
        oj = measurements.get("overjet_mm", 0.0)
        if oj > 3.0:
            recommendations.append(f"High overjet ({oj}mm) -> Initiate retraction mechanics.")
            
        # 4. Torque issues
        inclinations = measurements.get("buccolingual_inclination_degrees", {})
        for tooth, inc in inclinations.items():
            if abs(inc) > 5.0:
                recommendations.append(f"Torque issue on tooth {tooth} ({inc}°) -> Introduce finishing wire bend.")
                
        # 5. Marginal ridge discrepancy
        ridges = measurements.get("marginal_ridge_discrepancies_mm", {})
        for pair, diff in ridges.items():
            if diff > 0.5:
                recommendations.append(f"Marginal ridge discrepancy at {pair} ({diff}mm) -> Prescribe vertical settling elastics.")
                
        # Ensure we always return something
        if not recommendations:
            recommendations.append("Measurements indicate optimal alignment. Proceed with debonding protocols.")
            
        return recommendations
