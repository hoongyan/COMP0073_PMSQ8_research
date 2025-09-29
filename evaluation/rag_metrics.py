import sys
import os
import pandas as pd
import json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast  
import logging
from pathlib import Path

# Import settings and logger setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.settings import get_settings
from config.logging_config import setup_logger

class RagRelevanceCalculator:
    """Class for calculating RAG relevance metrics from CSV and JSON data."""
    def __init__(self, rag_csv_path: str, gt_json_path: str, history_csv_path: str = None):
        self.rag_csv_path = rag_csv_path
        self.gt_json_path = gt_json_path
        self.history_csv_path = history_csv_path
        self.rag_df = None
        self.history_df = None
        self.convo_to_profile = {}
        self.gt_data = None
        self.settings = get_settings()
        self.logger = setup_logger("rag_metrics", self.settings.log.subdirectories["evaluation"])

    def load_data(self):
        """Load and process data from CSV and JSON files."""
        # Load RAG invocation CSV
        self.rag_df = pd.read_csv(self.rag_csv_path, index_col=False, engine='python', on_bad_lines='warn')

        self.rag_df['conversation_id'] = pd.to_numeric(self.rag_df['conversation_id'], errors='coerce')
        self.rag_df = self.rag_df.dropna(subset=['conversation_id'])
        self.rag_df['conversation_id'] = self.rag_df['conversation_id'].astype('int')
        self.logger.info(f"Loaded RAG CSV with {len(self.rag_df)} rows.")
        self.logger.info(f"RAG columns: {self.rag_df.columns.tolist()}")
        self.logger.info(f"Sample scam_results (row 0): {self.rag_df['scam_results'].iloc[0] if not self.rag_df.empty else 'Empty'}")
        self.logger.info(f"Sample scam_distances (row 0): {self.rag_df['scam_distances'].iloc[0] if not self.rag_df.empty else 'Empty'}")

        if self.history_csv_path:
            self.history_df = pd.read_csv(self.history_csv_path, index_col=False, engine='python', on_bad_lines='warn')
            self.history_df['conversation_id'] = pd.to_numeric(self.history_df['conversation_id'], errors='coerce')
            self.history_df = self.history_df.dropna(subset=['conversation_id'])
            self.history_df['conversation_id'] = self.history_df['conversation_id'].astype('int')
            self.history_df['profile_id'] = pd.to_numeric(self.history_df['profile_id'], errors='coerce')
            self.history_df = self.history_df.dropna(subset=['profile_id'])
            self.history_df['profile_id'] = self.history_df['profile_id'].astype('int')
            self.logger.info(f"Loaded History CSV with {len(self.history_df)} rows.")

            grouped_history = self.history_df.groupby('conversation_id')
            for convo_id, group in grouped_history:
                profile_ids = group['profile_id'].unique()
                if len(profile_ids) == 1:
                    self.convo_to_profile[convo_id] = profile_ids[0]
                else:
                    self.logger.warning(f"Multiple profile_ids for convo {convo_id}: {profile_ids}. Skipping.")
            self.logger.info(f"Built mapping for {len(self.convo_to_profile)} conversations.")

        # Load GT JSON
        with open(self.gt_json_path, 'r') as f:
            gt_list = json.load(f)
        self.gt_data = {int(item['profile_id']): item for item in gt_list}
        self.logger.info(f"Loaded GT JSON with {len(self.gt_data)} entries.")

    def preprocess(self):
        """Preprocess the loaded data by grouping conversations and mapping to profiles."""
        # Group by conversation_id
        grouped = self.rag_df.groupby('conversation_id')
        preprocessed = {}
        for convo_id, group in grouped:
            if self.convo_to_profile and convo_id in self.convo_to_profile:
                profile_id = self.convo_to_profile[convo_id]
            else:
                profile_id = ((convo_id - 1) % 24) + 1
                self.logger.warning(f"No mapping for convo {convo_id}; using fallback modulo: {profile_id}")

            if profile_id not in self.gt_data:
                self.logger.warning(f"Skipping convo {convo_id}: Profile {profile_id} not in GT.")
                continue
            gt_scam_type = self.gt_data[profile_id]['scam_details']['scam_type'].strip().lower()
            gt_user_profile = self.gt_data[profile_id]['user_profile']

    
            group['timestamp'] = pd.to_datetime(group['timestamp'], errors='coerce')
            group = group.sort_values('timestamp')

            # Determine segment as full profile permutation (e.g., "low low distressed")
            tech = gt_user_profile.get('tech_literacy', 'unknown').lower()
            lang = gt_user_profile.get('language_proficiency', 'unknown').lower()
            emo = gt_user_profile.get('emotional_state', 'unknown').lower()
            seg = f"{tech} {lang} {emo}"

            preprocessed[convo_id] = {
                'profile_id': profile_id,
                'gt_scam_type': gt_scam_type,
                'gt_user_profile': gt_user_profile,
                'invocations': group.to_dict('records'),
                'segment': seg  
            }
        self.logger.info(f"Preprocessed {len(preprocessed)} conversations.")
        return preprocessed

    def compute_precision(self, scam_results_str: str, gt_scam_type: str) -> float:
        """Compute precision for retrieved scam results against ground truth scam type."""
        scam_results_str = str(scam_results_str).strip().strip('"').replace('""', '"')
        try:
            results = json.loads(scam_results_str)
        except:
            try:
                results = ast.literal_eval(scam_results_str)
            except:
                return 0.0
        if not isinstance(results, list) or not results:
            return 0.0
        relevant = sum(1 for res in results if isinstance(res, dict) and res.get('scam_type', '').strip().lower() == gt_scam_type)
        return relevant / len(results)

    def compute_metrics(self, preprocessed):
        """Compute aggregated precision metrics across models, segments, and types."""
        overall_precision = []
        by_model_precision = defaultdict(list)
        by_segment_precision = defaultdict(list)
        by_type_precision = defaultdict(list)
        by_model_segment_precision = defaultdict(lambda: defaultdict(list))
        by_segment_type_precision = defaultdict(lambda: defaultdict(list))  
        by_type_model_precision = defaultdict(lambda: defaultdict(list)) 
        per_convo_changes = {}

        for convo_id, data in preprocessed.items():
            gt_scam_type = data['gt_scam_type']
            seg = f"{data['gt_user_profile'].get('tech_literacy', 'unknown').lower()} {data['gt_user_profile'].get('language_proficiency', 'unknown').lower()} {data['gt_user_profile'].get('emotional_state', 'unknown').lower()}"

            precisions = []
            for inv in data['invocations']:
                scam_results_str = inv['scam_results']
                prec = self.compute_precision(scam_results_str, gt_scam_type)
                try:
                    results = json.loads(scam_results_str.replace('""', '"'))
                    retrieved_count = len(results) if isinstance(results, list) else 5
                except:
                    retrieved_count = 5
                relevant = int(prec * retrieved_count)
                precisions.append(prec)
                overall_precision.append(prec)
                model = inv.get('llm_model', 'unknown').strip().strip('\'"')
                by_model_precision[model].append(prec)
                by_segment_precision[seg].append(prec)
                by_type_precision[gt_scam_type].append(prec)
                by_model_segment_precision[model][seg].append(prec)
                by_segment_type_precision[seg][gt_scam_type].append(prec)  
                by_type_model_precision[gt_scam_type][model].append(prec)  
                self.logger.info(f"Invocation in convo {convo_id}: Prec={prec:.2f}, Relevant={relevant}/{retrieved_count}")

            if precisions:
                per_convo_changes[convo_id] = {
                    'initial_prec': precisions[0],
                    'last_prec': precisions[-1],
                    'improved': precisions[-1] > precisions[0] if precisions[0] < 1.0 else False
                }
                self.logger.info(f"Convo {convo_id}: Initial Prec={precisions[0]:.2f} -> Last={precisions[-1]:.2f}")

        def safe_mean(lst):
            return np.mean(lst) if lst else 0.0

        return {
            'overall_precision': safe_mean(overall_precision),
            'by_model_precision': {m: safe_mean(p) for m, p in by_model_precision.items()},
            'by_segment_precision': {s: safe_mean(p) for s, p in by_segment_precision.items()},
            'by_type_precision': {t: safe_mean(p) for t, p in by_type_precision.items()},
            'by_model_segment_precision': {m: {s: safe_mean(p) for s, p in segs.items()} for m, segs in by_model_segment_precision.items()},
            'by_segment_type_precision': {s: {t: safe_mean(p) for t, p in types.items()} for s, types in by_segment_type_precision.items()},  # NEW: Added to return dict
            'by_type_model_precision': {t: {m: safe_mean(p) for m, p in models.items()} for t, models in by_type_model_precision.items()},  # NEW: Added to return dict
            'per_convo_changes': per_convo_changes
        }

    def perform_eda(self, preprocessed):
        """Perform exploratory data analysis including distributions and confusion matrix."""
        gt_dist = Counter(data['gt_scam_type'] for data in preprocessed.values())
        retrieved_dist = Counter()
        
        # Compute averaged confusion matrix 
        confusion_accum = defaultdict(lambda: defaultdict(float)) 
        inv_count_by_type = defaultdict(int)  
        for data in preprocessed.values():
            gt_scam_type = data['gt_scam_type']
            for inv in data['invocations']:
                scam_results_str = str(inv['scam_results']).strip().strip('"').replace('""', '"')
                try:
                    results = json.loads(scam_results_str)
                    retrieved_types = []
                    for res in results:
                        if isinstance(res, dict):
                            retrieved_types.append(res.get('scam_type', 'unknown').strip().lower())
                    if not retrieved_types:
                        continue 
                    type_counter = Counter(retrieved_types)
                    retrieved_count = len(retrieved_types)

                    for ret_type, count in type_counter.items():
                        proportion = count / retrieved_count
                        confusion_accum[gt_scam_type][ret_type] += proportion
                    inv_count_by_type[gt_scam_type] += 1
                except Exception as e:
                    self.logger.warning(f"Confusion matrix parsing error: {e}")
                    continue  

        # Average the accumulated proportions across invocations per GT type
        confusion_avg = defaultdict(dict)
        for gt_type, ret_dict in confusion_accum.items():
            inv_count = inv_count_by_type[gt_type]
            if inv_count > 0:
                for ret_type, sum_prop in ret_dict.items():
                    confusion_avg[gt_type][ret_type] = sum_prop / inv_count

        self.logger.info(f"Averaged confusion matrix: {dict(confusion_avg)}")
        
        for data in preprocessed.values():
            for inv in data['invocations']:
                scam_results_str = str(inv['scam_results']).strip().strip('"').replace('""', '"')
                try:
                    results = json.loads(scam_results_str)
                    for res in results:
                        if isinstance(res, dict):
                            retrieved_dist[res.get('scam_type', 'unknown').strip().lower()] += 1
                except Exception as e:
                    self.logger.warning(f"EDA parsing error: {e}")
        self.logger.info(f"GT Scam Type Dist: {dict(gt_dist)}")
        self.logger.info(f"Retrieved Scam Type Dist: {dict(retrieved_dist)}")
        return {'gt_dist': gt_dist, 'retrieved_dist': retrieved_dist, 'confusion': confusion_avg} 

    def generate_charts(self, metrics, eda):
        """Generate and save visualization charts for metrics and EDA."""
        
        scam_type_map = {
            'government officials impersonation': 'GOIS',
            'ecommerce': 'ECOMMERCE',
            'phishing': 'PHISHING',
        }
        
        segment_map = {
            'high high neutral': 'HHN',
            'high high distressed': 'HHD',
            'high low neutral': 'HLN',
            'high low distressed': 'HLD',
            'low high neutral': 'LHN',
            'low high distressed': 'LHD',
            'low low neutral': 'LLN',
            'low low distressed': 'LLD',
        }
        
        desired_order = [
            'high high neutral',
            'high high distressed',
            'high low neutral',
            'high low distressed',
            'low high neutral',
            'low high distressed',
            'low low neutral',
            'low low distressed'
        ]
        
        if not metrics['by_model_precision']:
            self.logger.warning("No data for charts.")
            return

        # Set up output directory
        base_eval_dir = Path(self.settings.data.evaluation_results_dir)
        rag_dir = base_eval_dir / 'rag'
        rag_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Ensured output directory exists: {rag_dir}")
        models = [m.strip().strip("'\"") for m in metrics['by_model_precision'].keys()]
        prec_values = list(metrics['by_model_precision'].values())
        x = np.arange(len(models))
        width = 0.35
        fig, ax = plt.subplots()
        ax.bar(x, prec_values, width, label='Precision')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_title('Avg Precision by Model')
        for i, v in enumerate(prec_values):
            ax.text(i, v, f"{v:.2f}", ha='center', va='bottom')
        fig.savefig(rag_dir / 'rag_prec_by_model.png')
        plt.close(fig)

        if metrics['by_model_segment_precision']:
            df_prec = pd.DataFrame(metrics['by_model_segment_precision']).fillna(0)
            existing_segs = [s for s in desired_order if s in df_prec.index]
            df_prec = df_prec.reindex(existing_segs)
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(df_prec, annot=True, cmap='RdYlGn', ax=ax, vmin=0, vmax=1, annot_kws={"size": 25})
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=18)  
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', fontsize=18)
            cleaned_yticks = [label.get_text().strip("'\"") for label in ax.get_yticklabels()]
            mapped_yticks = [segment_map.get(tick, tick) for tick in cleaned_yticks]
            ax.set_yticklabels(mapped_yticks, rotation=0, ha='right', fontsize=18)
            
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=18)  
            ax.set_title('Precision by Model and Segment', fontsize=20)
            plt.tight_layout()
            fig.savefig(rag_dir / 'rag_prec_by_model_segment.png')
            plt.close(fig)

        types = list(metrics['by_type_precision'].keys())
        if types:
            prec_values = list(metrics['by_type_precision'].values())
            x = np.arange(len(types))
            fig, ax = plt.subplots()
            ax.bar(x, prec_values, width, label='Precision')
            ax.set_xticks(x)
            mapped_types = [scam_type_map.get(t, t) for t in types]  
            ax.set_xticklabels(mapped_types, rotation=0)
            ax.set_title('Avg Precision by Scam Type')
            for i, v in enumerate(prec_values):
                ax.text(i, v, f"{v:.2f}", ha='center', va='bottom')
            fig.savefig(rag_dir / 'rag_prec_by_type.png')
            plt.close(fig)
    
        # Bar chart for Avg Precision by Victim Segment
        if metrics['by_segment_precision']:
            segs = list(metrics['by_segment_precision'].keys())
            existing_segs = [s for s in desired_order if s in segs]
            ordered_precs = [metrics['by_segment_precision'][s] for s in existing_segs]
            mapped_segs = [segment_map.get(s, s) for s in existing_segs]
            
            x = np.arange(len(existing_segs))
            width = 0.35
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(x, ordered_precs, width, label='Precision')
            ax.set_xticks(x)
            ax.set_xticklabels(mapped_segs, rotation=0, ha='right', fontsize=12)
            ax.set_ylabel('Precision')
            ax.set_title('Avg Precision by Victim Segment')
            ax.set_ylim(0, 1.05) 
            for i, v in enumerate(ordered_precs):
                ax.text(i, v, f"{v:.2f}", ha='center', va='bottom')
            plt.tight_layout()
            fig.savefig(rag_dir / 'rag_prec_by_segment.png')
            plt.close(fig)     
    

        # Heatmap for Precision by Segment and Scam Type
        if metrics['by_segment_type_precision']:
            df_prec = pd.DataFrame(metrics['by_segment_type_precision']).T.fillna(0)
            # Reorder rows (segments) based on desired_order, keeping only existing segments
            existing_segs = [s for s in desired_order if s in df_prec.index]
            df_prec = df_prec.reindex(existing_segs)
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(df_prec, annot=True, cmap='RdYlGn', ax=ax, vmin=0, vmax=1, annot_kws={"size": 25})
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=18)
            mapped_yticks = [segment_map.get(label.get_text(), label.get_text()) for label in ax.get_yticklabels()]
            ax.set_yticklabels(mapped_yticks, rotation=0, ha='right', fontsize=18)
            mapped_xticks = [scam_type_map.get(label.get_text(), label.get_text()) for label in ax.get_xticklabels()]
            ax.set_xticklabels(mapped_xticks, rotation=0, fontsize=18)
            ax.set_title('Precision by Segment and Scam Type', fontsize=20)
            plt.tight_layout()
            fig.savefig(rag_dir / 'rag_prec_by_segment_type.png')
            plt.close(fig)

        # Heatmap for Precision by Scam Type and Model
        if metrics['by_type_model_precision']:
            df_prec = pd.DataFrame(metrics['by_type_model_precision']).fillna(0)
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(df_prec, annot=True, cmap='RdYlGn', ax=ax, vmin=0, vmax=1, annot_kws={"size": 25})
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=18)
            xticks = [label.get_text().strip().strip("'\"") for label in ax.get_xticklabels()]
            mapped_xticks = [scam_type_map.get(tick, tick) for tick in xticks]
            ax.set_xticklabels(mapped_xticks, rotation=0, fontsize=18)

            cleaned_yticks = [label.get_text().strip().strip("'\"") for label in ax.get_yticklabels()]
            ax.set_yticklabels(cleaned_yticks, rotation=0, ha='right', fontsize=18)
            plt.tight_layout()
            fig.savefig(rag_dir / 'rag_prec_by_type_model.png')
            plt.close(fig)
            
        # Averaged normalized confusion matrix heatmap 
        if 'confusion' in eda and eda['confusion']:
            confusion_df = pd.DataFrame(eda['confusion']).fillna(0).T
            confusion_df = confusion_df.reindex(sorted(confusion_df.columns), axis=1)
            confusion_df = confusion_df.reindex(sorted(confusion_df.index))
            mapped_columns = [scam_type_map.get(col, col) for col in confusion_df.columns]
            mapped_index = [scam_type_map.get(idx, idx) for idx in confusion_df.index]
            confusion_df.columns = mapped_columns
            confusion_df.index = mapped_index
            
            fig, ax = plt.subplots(figsize=(16, 12))
            sns.heatmap(confusion_df, annot=True, cmap='Blues', ax=ax, fmt='.2f', vmin=0, vmax=1, annot_kws={"size": 45})
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=35)
            ax.set_xlabel('RETRIEVED SCAM TYPE', fontsize=40)
            ax.set_ylabel('TRUE SCAM TYPE', fontsize=40)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=35, weight='bold')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=35, weight='bold')
            plt.tight_layout()
            fig.savefig(rag_dir / 'rag_confusion_matrix_normalized_per_inv.png')
            plt.close(fig)
            

        self.logger.info("Generated charts.")

    def analyze_changes(self, metrics):
        """Analyze changes in precision over conversations."""
        improved_count = sum(1 for v in metrics['per_convo_changes'].values() if v['improved'])
        total_with_issue = sum(1 for v in metrics['per_convo_changes'].values() if v['initial_prec'] < 1.0)
        self.logger.info(f"Convos improved (prec): {improved_count}/{total_with_issue} (for conversations that did not start with perfect precision)")

    def run(self):
        """Run the full calculation process: load, preprocess, EDA, metrics, charts, and analysis."""
        self.load_data()
        preprocessed = self.preprocess()
        eda = self.perform_eda(preprocessed)
        metrics = self.compute_metrics(preprocessed)
        self.generate_charts(metrics, eda)
        self.analyze_changes(metrics)
        return metrics


if __name__ == "__main__":

    rag_csv = "simulations/final_results/phase_2/profile_rag_ie_kb/autonomous_rag_invocation.csv"
    history_csv = "simulations/final_results/phase_2/profile_rag_ie_kb/autonomous_conversation_history.csv"
    victim_detail_json = "data/victim_profile/victim_details.json"

    calculator = RagRelevanceCalculator(rag_csv, victim_detail_json, history_csv)
    results = calculator.run()
    print(results)  
