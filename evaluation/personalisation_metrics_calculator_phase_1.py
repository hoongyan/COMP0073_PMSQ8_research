import sys
import os
import pandas as pd
import json
from collections import defaultdict
import numpy as np
import logging
from pathlib import Path

class IeScoreCalculator:
    def __init__(self, ie_csv_path: str, gt_json_path: str):
        self.ie_csv_path = ie_csv_path
        self.gt_json_path = gt_json_path
        self.ie_df = None
        self.gt_data = None
        self.logger = logging.getLogger("ie_metrics")
        logging.basicConfig(level=logging.INFO)

    def load_data(self):
        # Load IE CSV
        self.ie_df = pd.read_csv(self.ie_csv_path, index_col=False)
        # Force types for conversation_id
        self.ie_df['conversation_id'] = pd.to_numeric(self.ie_df['conversation_id'], errors='coerce')
        self.ie_df = self.ie_df.dropna(subset=['conversation_id'])
        self.ie_df['conversation_id'] = self.ie_df['conversation_id'].astype('int')
        
        # NEW: Force types for profile_id (since it's now directly in the CSV)
        # This ensures it's an integer, dropping any bad rows
        self.ie_df['profile_id'] = pd.to_numeric(self.ie_df['profile_id'], errors='coerce')
        self.ie_df = self.ie_df.dropna(subset=['profile_id'])
        self.ie_df['profile_id'] = self.ie_df['profile_id'].astype('int')
        
        self.logger.info(f"Loaded IE CSV with {len(self.ie_df)} rows.")
        self.logger.info(f"IE columns: {self.ie_df.columns.tolist()}")

        # Load GT JSON (keyed by int profile_id)
        with open(self.gt_json_path, 'r') as f:
            gt_list = json.load(f)
        self.gt_data = {int(item['profile_id']): item for item in gt_list}
        self.logger.info(f"Loaded GT JSON with {len(self.gt_data)} entries.")

    def preprocess(self):
        # Group by conversation_id
        grouped = self.ie_df.groupby('conversation_id')
        preprocessed = {}
        for convo_id, group in grouped:
            # CHANGED: Instead of computing profile_id with modulo, get it directly from the CSV group
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

            # Compute average avg_rating_normalized for this conversation
            avg_score = group['avg_rating_normalized'].mean()

            # Get LLM model from IE CSV group
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
        self.load_data()
        preprocessed = self.preprocess()
        metrics = self.compute_metrics(preprocessed)
        return metrics


if __name__ == "__main__":
    ie_csv = "evaluation/results/communication_appropriateness/phase_1/evaluated_autonomous_conversations_rag_ie.csv"  # Replace with actual path
    victim_detail_json = "data/victim_profile/victim_details_human_eval.json"  # Replace with actual path
    
    calculator = IeScoreCalculator(ie_csv, victim_detail_json)
    results = calculator.run()
    print(results)  # Output metrics