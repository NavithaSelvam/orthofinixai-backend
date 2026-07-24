import sys
import os

# Add backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.ai_models.clinical_analysis.clinical_measurement_engine import ClinicalMeasurementEngine
from app.services.ai_models.clinical_analysis.abo_scoring_engine import AboScoringEngine
from app.services.ai_models.clinical_analysis.andrews_evaluation_engine import AndrewsEvaluationEngine
from app.services.ai_models.clinical_analysis.roling_evaluation_engine import RolingEvaluationEngine

def test_engine():
    # Mock landmarks
    landmarks = {
        "11_incisal_edge": (0.50, 0.45),
        "21_incisal_edge": (0.55, 0.45),
        "31_incisal_edge": (0.50, 0.48),
        "41_incisal_edge": (0.55, 0.48),
        "11_cej_mesial": (0.49, 0.40),
        "11_cej_distal": (0.51, 0.40),
        "21_cej_mesial": (0.54, 0.40),
        "21_cej_distal": (0.56, 0.40),
        "31_cej_mesial": (0.51, 0.52),
        "41_cej_mesial": (0.54, 0.52),
        "16_cusp_tip_buccal": (0.20, 0.42),
        "26_cusp_tip_buccal": (0.80, 0.42),
        "47_cusp_tip_buccal": (0.82, 0.55),
        "46_fa": (0.80, 0.53),
        "47_fa": (0.82, 0.56),
        "11_fa": (0.50, 0.42),
        "11_apex": (0.50, 0.35)
    }

    engine = ClinicalMeasurementEngine(scale_factor=100.0) # Assume 1.0 normalized = 100 mm for easy reading
    results = engine.analyze(landmarks)
    
    for key, value in results.items():
        print(f"{key}: {value}")

    print("\n--- ABO Scoring Engine ---")
    abo_engine = AboScoringEngine()
    abo_results = abo_engine.score(results)
    
    import json
    print(json.dumps(abo_results, indent=2))

    print("\n--- Andrews Six Keys ---")
    andrews = AndrewsEvaluationEngine()
    print(json.dumps(andrews.evaluate(results), indent=2))

    print("\n--- Rebecca Roling Evaluation ---")
    roling = RolingEvaluationEngine()
    print(json.dumps(roling.evaluate(results), indent=2))

if __name__ == "__main__":
    test_engine()
