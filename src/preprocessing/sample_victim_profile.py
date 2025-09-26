import sys
import os
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config.settings import get_settings  

def sample_profiles(input_path: str, output_path: str, num_per_combination: int = 1) -> None:
    """
    Sample victim profiles from the victim_details.json file to compile the following high and low risk profiles (extreme ends of the spectrum):
    {"tech_literacy": "high", "language_proficiency": "high","emotional_state": "neutral"} and {"tech_literacy": "high", "language_proficiency": "high","emotional_state": "neutral"}
    The sampled set of profiles will be used for evaluation in Phase 1 (human evaluation).
    """
    # Load the JSON data
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Group entries by scam_type and risk level
    groups = defaultdict(list)
    for entry in data:
        scam_type = entry['scam_details']['scam_type'].lower()
        
        profile = entry['user_profile']
        if (profile['tech_literacy'] == 'low' and
            profile['language_proficiency'] == 'low' and
            profile['emotional_state'] == 'distressed'):
            risk = 'high'
        elif (profile['tech_literacy'] == 'high' and
              profile['language_proficiency'] == 'high' and
              profile['emotional_state'] == 'neutral'):
            risk = 'low'
        else:
            continue  
        groups[(scam_type, risk)].append(entry)
    
    # Define the 3 scam types
    scam_types = ['ecommerce', 'phishing', 'government officials impersonation']
    
    # Sample the specified number per combination (1 per combination)
    sampled = []
    for scam_type in scam_types:
        for risk in ['high', 'low']:
            group_key = (scam_type, risk)
            if groups[group_key]:
                # Randomly sample without replacement if num_per_combination > 1 - for scalability
                sampled.extend(random.sample(groups[group_key], min(num_per_combination, len(groups[group_key]))))
            else:
                print(f"Warning: No entries found for combination {scam_type} - {risk}")
    
    # Save sampled data
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(sampled, f, indent=2)
    
    print(f"Sampled {len(sampled)} entries and saved to {output_path}")

if __name__ == "__main__":
    settings = get_settings()  # Load centralized settings
    
    parser = argparse.ArgumentParser(description="Sample victim profiles from JSON for evaluation.")
    parser.add_argument("--input_path", type=str, default=settings.data.victim_details_json, help="Path to input JSON file.")
    parser.add_argument("--output_path", type=str, default="data/victim_profile/victim_detail_human_eval.json", help="Path to output sampled JSON file.")
    parser.add_argument("--num_per_combination", type=int, default=1, help="Number of samples per risk-scam_type combination.")
    
    args = parser.parse_args()
    sample_profiles(args.input_path, args.output_path, args.num_per_combination)