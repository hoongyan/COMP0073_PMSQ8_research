import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..','..')))

from typing import Dict, List
from src.models.response_model import TechLiteracy, LanguageProficiency, EmotionalState
import logging
import pandas as pd 
from itertools import product

logger = logging.getLogger(__name__)
    
class PersonaGenerator:
    """Subclass for generating all persona permutations to create victim_details.json for evaluation."""
    
    def __init__(self):
        pass

    def generate_all_permutation_targets(self) -> List[Dict[str, str]]:
        """Generate list of all 8 target persona permutations as dicts."""
        permutations = product(
            [e.value for e in TechLiteracy],
            [e.value for e in LanguageProficiency],
            [e.value for e in EmotionalState]
        )
        return [{"tech_literacy": t, "language_proficiency": l, "emotional_state": e} for t, l, e in permutations]

    def generate_high_risk_profile(self) -> Dict[str,str]:
        """Generate high-risk profile consisting of low tech_literacy, low language_proficiency and distressed emotional_state."""
        
        return {"tech_literacy": TechLiteracy.low.value, 
                "language_proficiency": LanguageProficiency.low.value, 
                "emotional_state": EmotionalState.distressed.value}
    
    def generate_low_risk_profile(self) -> Dict[str,str]:
        """Generate low-risk profile consisting of high tech_literacy, high language_proficiency and neutral emotional_state"""
        return {"tech_literacy": TechLiteracy.high.value, 
                "language_proficiency": LanguageProficiency.high.value, 
                "emotional_state": EmotionalState.neutral.value}
    
#Test code 
if __name__ == "__main__":

    persona = PersonaGenerator()  
    persona_df = persona.generate_all_permutation_targets()
    print(persona_df)
    low_risk = persona.generate_low_risk_profile()
    print(low_risk)
    high_risk=persona.generate_high_risk_profile()
    print(high_risk)