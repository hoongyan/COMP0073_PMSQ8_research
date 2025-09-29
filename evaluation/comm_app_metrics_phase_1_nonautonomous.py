import sys
import os
import pandas as pd
import json
from collections import defaultdict
import numpy as np
import logging
from pathlib import Path
import ast

class CommAppScoreCalculator:
    """Class for calculating communication appropriateness scores based on CSV and JSON data."""
    def __init__(self, ie_csv_path: str, gt_json_path: str):
        self.ie_csv_path = ie_csv_path
        self.gt_json_path = gt_json_path
        self.ie_df = None
        self.gt_data = None
        self.logger = logging.getLogger("ie_metrics")
        logging.basicConfig(level=logging.INFO)

    def load_data(self):
        """Load data from the CSV and ground truth JSON files."""
        # Load CSV
        self.ie_df = pd.read_csv(self.ie_csv_path, index_col=False)

        self.ie_df['conversation_id'] = pd.to_numeric(self.ie_df['conversation_id'], errors='coerce')
        self.ie_df = self.ie_df.dropna(subset=['conversation_id'])
        self.ie_df['conversation_id'] = self.ie_df['conversation_id'].astype('int')

        self.ie_df['profile_id'] = pd.to_numeric(self.ie_df['profile_id'], errors='coerce')
        self.ie_df = self.ie_df.dropna(subset=['profile_id'])
        self.ie_df['profile_id'] = self.ie_df['profile_id'].astype('int')
        
        self.logger.info(f"Loaded IE CSV with {len(self.ie_df)} rows.")
        self.logger.info(f"IE columns: {self.ie_df.columns.tolist()}")

        # Load GT JSON
        with open(self.gt_json_path, 'r') as f:
            gt_list = json.load(f)
        self.gt_data = {int(item['profile_id']): item for item in gt_list}
        self.logger.info(f"Loaded GT JSON with {len(self.gt_data)} entries.")

    def preprocess(self):
        """Preprocess the loaded data by grouping conversations and extracting relevant info."""
        # Group by conversation_id
        grouped = self.ie_df.groupby('conversation_id')
        preprocessed = {}
        for convo_id, group in grouped:
      
            profile_id = int(group['profile_id'].iloc[0])
            
            if profile_id not in self.gt_data:
                self.logger.warning(f"Skipping convo {convo_id}: Computed profile {profile_id} not in GT.")
                continue
            gt_user_profile = self.gt_data[profile_id]['user_profile']

            # Determine segment as full profile permutation (e.g., "low low distressed")
            tech = gt_user_profile.get('tech_literacy', 'unknown').lower()
            lang = gt_user_profile.get('language_proficiency', 'unknown').lower()
            emo = gt_user_profile.get('emotional_state', 'unknown').lower()
            seg = f"{tech} {lang} {emo}"

            # Find police rows with communication_appropriateness_rating
            police_rows = group[(group['sender_type'] == 'police') & (group['communication_appropriateness_rating'].notna())]

            scores = []
            for _, row in police_rows.iterrows():
                persona_str = row['communication_appropriateness_rating']
                if persona_str and persona_str != '{}':
                    try:
                        persona_dict = ast.literal_eval(persona_str)
                        turn_scores = [
                            persona_dict.get('language_proficiency'),
                            persona_dict.get('tech_literacy'),
                            persona_dict.get('emotional_state')
                        ]
                        turn_scores = [s for s in turn_scores if s is not None and not np.isnan(s)]
                        if turn_scores:
                            turn_avg = np.mean(turn_scores)
                            scores.append(turn_avg)
                    except Exception as e:
                        self.logger.warning(f"Error parsing persona dict in convo {convo_id}: {e}")

            if scores:
                avg_score = np.mean(scores) / 5.0  # Normalize to 0-1
            else:
                avg_score = np.nan

            # Get LLM model 
            llm_model = group['police_llm_model'].iloc[0]

            preprocessed[convo_id] = {
                'profile_id': profile_id,
                'gt_user_profile': gt_user_profile,
                'avg_score': avg_score,
                'segment': seg,
                'llm_model': llm_model
            }
        self.logger.info(f"Preprocessed {len(preprocessed)} conversations.")
        return preprocessed

    def compute_metrics(self, preprocessed):
        """Compute overall and grouped metrics from preprocessed data."""
        
        # Collect all per-convo avg_scores
        all_scores = [data['avg_score'] for data in preprocessed.values() if not np.isnan(data['avg_score'])]

        # Overall mean and std
        overall_mean = np.mean(all_scores) if all_scores else 0.0
        overall_std = np.std(all_scores) if all_scores else 0.0

        # By segment
        by_segment_scores = defaultdict(list)
        for data in preprocessed.values():
            if not np.isnan(data['avg_score']):
                by_segment_scores[data['segment']].append(data['avg_score'])

        by_segment_mean_std = {
            seg: {
                'mean': np.mean(scores) if scores else 0.0,
                'std': np.std(scores) if scores else 0.0,
                'count': len(scores)
            } for seg, scores in by_segment_scores.items()
        }

        # By LLM model
        by_llm_scores = defaultdict(list)
        for data in preprocessed.values():
            if not np.isnan(data['avg_score']):
                by_llm_scores[data['llm_model']].append(data['avg_score'])

        by_llm_mean_std = {
            llm: {
                'mean': np.mean(scores) if scores else 0.0,
                'std': np.std(scores) if scores else 0.0,
                'count': len(scores)
            } for llm, scores in by_llm_scores.items()
        }

        # By LLM model and segment
        by_llm_segment_scores = defaultdict(lambda: defaultdict(list))
        for data in preprocessed.values():
            if not np.isnan(data['avg_score']):
                by_llm_segment_scores[data['llm_model']][data['segment']].append(data['avg_score'])

        by_llm_segment_mean_std = {
            llm: {
                seg: {
                    'mean': np.mean(scores) if scores else 0.0,
                    'std': np.std(scores) if scores else 0.0,
                    'count': len(scores)
                } for seg, scores in segments.items()
            } for llm, segments in by_llm_segment_scores.items()
        }

        return {
            'overall': {
                'mean': overall_mean,
                'std': overall_std,
                'count': len(all_scores)
            },
            'by_segment': by_segment_mean_std,
            'by_llm': by_llm_mean_std,
            'by_llm_segment': by_llm_segment_mean_std
        }

    def run(self):
        """Run the full pipeline: load data, preprocess, and compute metrics."""
        self.load_data()
        preprocessed = self.preprocess()
        metrics = self.compute_metrics(preprocessed)
        return metrics


if __name__ == "__main__":
    ie_csv = "simulations/final_results/phase_1/rag_ie/autonomous_conversation_history_rag_ie_evaluated.csv"  
    victim_detail_json = "data/victim_profile/victim_details_human_eval.json"  
    
    calculator = CommAppScoreCalculator(ie_csv, victim_detail_json)
    results = calculator.run()
    print(results)  # Output metrics