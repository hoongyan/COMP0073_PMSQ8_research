import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.logging_config import setup_logger
from config.settings import get_settings
import pandas as pd
import json
import ast
import re
import math
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer, util
from sklearn.linear_model import LinearRegression
import matplotlib.colors as mcolors

class ScamAccuracyCalculator:
    def __init__(self, csv_path: str, json_path: str):
        """
        Initialize with file paths.
        """
        self.csv_path = csv_path
        self.json_path = json_path
        self.df = None
        self.ground_truth_scam = None
        self.ground_truth_profile = None
        self.settings = get_settings()
        self.logger = setup_logger("ie", self.settings.log.subdirectories["evaluation"])
        self.sim_model = SentenceTransformer('all-MiniLM-L6-v2')

    def load_data(self):
        """
        Load CSV and JSON data.
        """
        try:
            self.df = pd.read_csv(self.csv_path)
            self.logger.info(f"Loaded CSV with {len(self.df)} rows.")
        except FileNotFoundError:
            self.logger.error(f"CSV file not found: {self.csv_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading CSV: {e}")
            raise
        try:
            with open(self.json_path, 'r') as f:
                ground_truth_list = json.load(f)
            self.ground_truth_scam = {item['profile_id']: item['scam_details'] for item in ground_truth_list}
            self.ground_truth_profile = {item['profile_id']: item['user_profile'] for item in ground_truth_list}
            self.logger.info(f"Loaded JSON with {len(ground_truth_list)} entries.")
        except FileNotFoundError:
            self.logger.error(f"JSON file not found: {self.json_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading JSON: {e}")
            raise

    def preprocess_csv(self):
        """
        Preprocess CSV: Convert timestamps, group by conversation_id, take last row.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], errors='coerce')
        last_rows = self.df.loc[self.df.groupby('conversation_id')['timestamp'].idxmax()].set_index('conversation_id')
        self.logger.info(f"Preprocessed CSV to {len(last_rows)} unique conversations.")
        return last_rows

    def parse_scam_details(self, pred_str):
        """
        Parse scam_details string to dict.
        """
        if pd.isna(pred_str) or not pred_str.strip():
            return {}
        try:
            return ast.literal_eval(pred_str)
        except (ValueError, SyntaxError) as e:
            self.logger.warning(f"Error parsing scam_details: {e}. Returning empty dict.")
            return {}

    def normalize_field(self, key, val):
        """
        Field-specific normalization.
        """
        if val is None:
            val = ''
        val = str(val).strip()
        null_synonyms = {'', 'na', 'n/a', 'none', 'not applicable', 'not available', 'not provided', 'unknown'}
        if val.lower() in null_synonyms:
            return 'NA'
        if key == 'scam_contact_no':
            val = re.sub(r'\D', '', val)
            if val.startswith('65'):
                val = val[2:]
            elif val.startswith('+65'):
                val = val[3:]
            elif val.startswith('+'):
                val = val[1:]
            return val[-8:] if val else 'NA'
        elif key == 'scam_beneficiary_identifier':
            val = re.sub(r'\D', '', val)
            return val if val else 'NA'
        elif key == 'scam_url_link':
            val = re.sub(r'<[^>]*>', '', val)
            val = re.sub(r'^(https?://|www\.)', '', val.lower())
            val = val.rstrip('/')
            return val if val else 'NA'
        elif key == 'scam_amount_lost':
            val = val.replace('$', '').replace(',', '')
            try:
                return float(val)
            except ValueError:
                return 0.0
        elif key == 'scam_incident_date':
            try:
                dt = pd.to_datetime(val)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                return val
        uppercase_keys = [
            'scam_type', 'scam_approach_platform', 'scam_communication_platform',
            'scam_transaction_type', 'scam_beneficiary_platform'
        ]
        lowercase_keys = ['scam_moniker', 'scam_email']
        if key in uppercase_keys:
            return val.upper()
        elif key in lowercase_keys:
            return val.lower()
        else:
            return val.lower()

    def normalize_profile_value(self, val):
        """
        Normalize profile values.
        """
        if val is None:
            return 'na'
        val = str(val).strip().lower()
        null_synonyms = {'', 'na', 'n/a', 'none', 'not applicable', 'not available', 'not provided', 'unknown'}
        if val in null_synonyms:
            return 'na'
        return val

    def compare_details(self, pred_dict, gt_dict):
        """
        Compare normalized predicted dict to ground truth.
        """
        exclude = ['scam_incident_description', 'scam_report_no', 'scam_report_date']
        keys_to_compare = [k for k in gt_dict if k not in exclude]
        correct = 0
        total = len(keys_to_compare)
        per_field_correct = {k: 0 for k in keys_to_compare}
        mismatches = []
        
        for key in keys_to_compare:
            if key not in pred_dict:
                raw_gt = gt_dict.get(key, 'NA')
                norm_gt = self.normalize_field(key, raw_gt)
                mismatches.append((key, raw_gt, norm_gt, 'missing', 'missing'))
                continue
            
            raw_gt = gt_dict[key]
            raw_pred = pred_dict[key]
            norm_gt = self.normalize_field(key, raw_gt)
            norm_pred = self.normalize_field(key, raw_pred)
            
            match = False

            if key == 'scam_amount_lost':
                if isinstance(norm_pred, (int, float)) and isinstance(norm_gt, (int, float)):
                    match = math.isclose(norm_pred, norm_gt, abs_tol=0.01)
                else:
                    match = norm_pred == norm_gt
            elif key == 'scam_incident_date':
                match = norm_pred == norm_gt
            else:
                match = norm_pred == norm_gt
            
            if match:
                correct += 1
                per_field_correct[key] = 1
            else:
                mismatches.append((key, raw_gt, norm_gt, raw_pred, norm_pred))
        
        return correct, total, per_field_correct, mismatches

    def evaluate_description(self, pred_dict, gt_dict):
        """
        Evaluate scam_incident_description: semantic similarity only.
        """
        if 'scam_incident_description' not in pred_dict or not pred_dict['scam_incident_description'].strip():
            return 0.0
        pred_desc = pred_dict['scam_incident_description']
        gt_desc = gt_dict.get('scam_incident_description', '')
        sim = 0.0
        if gt_desc.strip():
            embed_gt = self.sim_model.encode(gt_desc)
            embed_pred = self.sim_model.encode(pred_desc)
            sim = util.cos_sim(embed_gt, embed_pred)[0][0].item()
        return sim

    def compute_accuracies(self, last_rows):
        """
        Compute all accuracies and description similarities.
        """
        per_convo_accuracies = {}
        overall_correct = 0
        overall_total = 0
        
        by_police_model = defaultdict(lambda: {'correct': 0, 'total': 0})
        by_permutation = defaultdict(lambda: {'correct': 0, 'total': 0})
        by_police_model_and_permutation = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
        
        overall_per_field_correct = defaultdict(int)
        overall_per_field_total = defaultdict(int)
        by_model_per_field = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
        by_permutation_per_field = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
        by_model_permutation_per_field = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0})))
        by_model_scam_type = defaultdict(lambda: defaultdict(lambda: {'sum_acc': 0.0, 'count': 0}))
        
        categorical_fields = [
            'scam_type', 'scam_approach_platform', 'scam_communication_platform',
            'scam_transaction_type', 'scam_beneficiary_platform'
        ]
        confusion_per_field = {key: defaultdict(lambda: defaultdict(int)) for key in categorical_fields}
        
        all_mismatches = []
        all_sims = []
        overall_sum_sim = 0.0
        num_convos = 0
        
        by_model_sum_sim = defaultdict(float)
        by_model_num = defaultdict(int)
        
        by_permutation_sum_sim = defaultdict(float)
        by_permutation_num = defaultdict(int)
        
        by_model_permutation_sum_sim = defaultdict(lambda: defaultdict(float))
        by_model_permutation_num = defaultdict(lambda: defaultdict(int))
        
        
        for convo_id, row in last_rows.iterrows():
            profile_id = row['profile_id']
            if profile_id not in self.ground_truth_scam:
                self.logger.warning(f"Profile ID {profile_id} for convo {convo_id} not in ground truth. Skipping.")
                continue
            
            gt_dict = self.ground_truth_scam[profile_id]
            pred_dict = self.parse_scam_details(row['scam_details'])
            
            correct, total, per_field_correct, mismatches = self.compare_details(pred_dict, gt_dict)
            accuracy = correct / total if total > 0 else 0.0
            
            per_convo_accuracies[convo_id] = {
                'profile_id': profile_id,
                'police_llm_model': row['police_llm_model'],
                'accuracy': accuracy,
                'correct': correct,
                'total': total
            }
            
            overall_correct += correct
            overall_total += total
            
            model = row['police_llm_model']
            by_police_model[model]['correct'] += correct
            by_police_model[model]['total'] += total
            
            # Moved here: Update by_model_scam_type (now after model is defined)
            gt_scam_type = self.normalize_field('scam_type', gt_dict.get('scam_type', 'UNKNOWN'))
            by_model_scam_type[model][gt_scam_type]['sum_acc'] += accuracy
            by_model_scam_type[model][gt_scam_type]['count'] += 1
            
            user_prof = self.ground_truth_profile.get(profile_id, {})
            tl = self.normalize_profile_value(user_prof.get('tech_literacy', 'na'))
            lp = self.normalize_profile_value(user_prof.get('language_proficiency', 'na'))
            es = self.normalize_profile_value(user_prof.get('emotional_state', 'na'))
            permutation = f"{tl}_{lp}_{es}"
            
            by_permutation[permutation]['correct'] += correct
            by_permutation[permutation]['total'] += total
            
            by_police_model_and_permutation[model][permutation]['correct'] += correct
            by_police_model_and_permutation[model][permutation]['total'] += total
            
            for key in gt_dict:
                if key in ['scam_incident_description', 'scam_report_no', 'scam_report_date']:
                    continue
                overall_per_field_total[key] += 1
                by_model_per_field[model][key]['total'] += 1
                by_permutation_per_field[permutation][key]['total'] += 1
                by_model_permutation_per_field[model][permutation][key]['total'] += 1
                
                if key in per_field_correct and per_field_correct[key]:
                    overall_per_field_correct[key] += 1
                    by_model_per_field[model][key]['correct'] += 1
                    by_permutation_per_field[permutation][key]['correct'] += 1
                    by_model_permutation_per_field[model][permutation][key]['correct'] += 1
            
            for key in categorical_fields:
                if key in gt_dict:
                    norm_gt = self.normalize_field(key, gt_dict[key])
                    if key in pred_dict:
                        norm_pred = self.normalize_field(key, pred_dict[key])
                    else:
                        norm_pred = 'missing'
                    confusion_per_field[key][norm_gt][norm_pred] += 1
            
            for m in mismatches:
                all_mismatches.append((convo_id, model, permutation) + m)
            
            sim = self.evaluate_description(pred_dict, gt_dict)
            all_sims.append(sim)
            overall_sum_sim += sim
            num_convos += 1
            
            by_model_sum_sim[model] += sim
            by_model_num[model] += 1
            
            by_permutation_sum_sim[permutation] += sim
            by_permutation_num[permutation] += 1
            
            by_model_permutation_sum_sim[model][permutation] += sim
            by_model_permutation_num[model][permutation] += 1
            
            per_convo_accuracies[convo_id]['desc_similarity'] = sim
        
        overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0.0
        
        by_police_model_acc = {m: data['correct'] / data['total'] if data['total'] > 0 else 0.0
                              for m, data in by_police_model.items()}
        by_permutation_acc = {s: data['correct'] / data['total'] if data['total'] > 0 else 0.0
                             for s, data in by_permutation.items()}
        by_police_model_and_permutation_acc = {}
        for model, perm_dict in by_police_model_and_permutation.items():
            by_police_model_and_permutation_acc[model] = {s: data['correct'] / data['total'] if data['total'] > 0 else 0.0
                                                         for s, data in perm_dict.items()}
        
        overall_per_field_acc = {f: overall_per_field_correct[f] / overall_per_field_total[f] if overall_per_field_total[f] > 0 else 0.0
                                for f in overall_per_field_total}
        by_model_per_field_acc = {}
        for model, field_dict in by_model_per_field.items():
            by_model_per_field_acc[model] = {f: data['correct'] / data['total'] if data['total'] > 0 else 0.0
                                            for f, data in field_dict.items()}
        by_permutation_per_field_acc = {}
        for perm, field_dict in by_permutation_per_field.items():
            by_permutation_per_field_acc[perm] = {f: data['correct'] / data['total'] if data['total'] > 0 else 0.0
                                                 for f, data in field_dict.items()}
        by_model_permutation_per_field_acc = {}
        for model, perm_dict in by_model_permutation_per_field.items():
            by_model_permutation_per_field_acc[model] = {}
            for perm, field_dict in perm_dict.items():
                by_model_permutation_per_field_acc[model][perm] = {f: data['correct'] / data['total'] if data['total'] > 0 else 0.0
                                                                  for f, data in field_dict.items()}
        
        by_model_scam_type_acc = {}
        for model, scam_dict in by_model_scam_type.items():
            by_model_scam_type_acc[model] = {s: data['sum_acc'] / data['count'] if data['count'] > 0 else 0.0
                                            for s, data in scam_dict.items()}
            
        overall_desc_similarity = overall_sum_sim / num_convos if num_convos > 0 else 0.0
        
        by_model_desc_similarity = {m: by_model_sum_sim[m] / by_model_num[m] if by_model_num[m] > 0 else 0.0
                                   for m in by_model_sum_sim}
        
        by_permutation_desc_similarity = {p: by_permutation_sum_sim[p] / by_permutation_num[p] if by_permutation_num[p] > 0 else 0.0
                                         for p in by_permutation_sum_sim}
        
        by_model_permutation_desc_similarity = {}
        for model, perm_dict in by_model_permutation_sum_sim.items():
            by_model_permutation_desc_similarity[model] = {p: by_model_permutation_sum_sim[model][p] / by_model_permutation_num[model][p] if by_model_permutation_num[model][p] > 0 else 0.0
                                                          for p in perm_dict}
        
        return {
            'per_conversation': per_convo_accuracies,
            'overall_accuracy': overall_accuracy,
            'by_police_model': by_police_model_acc,
            'by_victim_permutation': by_permutation_acc,
            'by_police_model_and_permutation': by_police_model_and_permutation_acc,
            'overall_per_field_acc': overall_per_field_acc,
            'by_model_per_field_acc': by_model_per_field_acc,
            'by_permutation_per_field_acc': by_permutation_per_field_acc,
            'by_model_permutation_per_field_acc': by_model_permutation_per_field_acc,
            'confusion_per_field': confusion_per_field,
            'all_mismatches': all_mismatches,
            'all_desc_similarities': all_sims,
            'overall_desc_similarity': overall_desc_similarity,
            'by_model_desc_similarity': by_model_desc_similarity,
            'by_permutation_desc_similarity': by_permutation_desc_similarity,
            'by_model_permutation_desc_similarity': by_model_permutation_desc_similarity,
            'by_model_scam_type_acc': by_model_scam_type_acc
        }

    def compute_per_turn_accuracies(self):
        """
        Compute average accuracy per police turn position per model.
        """
        grouped = self.df[self.df['sender_type'] == 'police'].groupby('conversation_id')
        per_turn_data = defaultdict(lambda: defaultdict(list))
        for convo_id, group in grouped:
            profile_id = group['profile_id'].iloc[0]
            if profile_id not in self.ground_truth_scam:
                continue
            gt_dict = self.ground_truth_scam[profile_id]
            model = group['police_llm_model'].iloc[0]
            group = group.sort_values('turn_count')
            for _, row in group.iterrows():
                turn = row['turn_count']
                pred_dict = self.parse_scam_details(row['scam_details'])
                correct, total, _, _ = self.compare_details(pred_dict, gt_dict)
                acc = correct / total if total > 0 else 0.0
                per_turn_data[model][turn].append(acc)
        models = sorted(per_turn_data.keys())
        avg_per_turn_by_model = {model: [] for model in models}
        count_per_turn_by_model = {model: [] for model in models}
        max_turn = max(max(per_turn_data[model].keys()) for model in models if per_turn_data[model]) if models else 0
        for model in models:
            for turn in range(max_turn + 1):
                accs = per_turn_data[model].get(turn, [])
                if accs:
                    avg_per_turn_by_model[model].append(np.mean(accs))
                    count_per_turn_by_model[model].append(len(accs))
                else:
                    avg_per_turn_by_model[model].append(np.nan)
                    count_per_turn_by_model[model].append(0)
        return avg_per_turn_by_model, count_per_turn_by_model, max_turn

    def save_figure(self, fig, filepath: str) -> None:
        """
        Saves a Matplotlib figure to the given filepath.
        """
        try:
            fig.savefig(filepath)
            self.logger.info(f"Saved figure to '{filepath}'")
        except Exception as e:
            self.logger.error(f"Failed to save figure to '{filepath}': {e}")
            raise
        finally:
            plt.close(fig)

    def generate_charts(self, results, per_convo_accuracies, avg_per_turn_by_model, count_per_turn_by_model, max_turn):
        """
        Generate visualizations with updated requirements.
        """
        # Helper function to clean field labels
        def clean_label(label):
            if label == 'scam_type':
                return 'scam_type' 
            label = label.replace('scam_', '')  
            label = label.replace('communication_platform', 'comm_platform')
            return label.replace('GOVERNMENT OFFICIALS IMPERSONATION', 'GOIS')

        base_eval_dir = Path(self.settings.data.evaluation_results_dir)
        scam_details_dir = base_eval_dir / 'scam_details'
        scam_details_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Ensured output directory exists: {scam_details_dir}")

        #Bar chart for per-field accuracies (structured)
        fields = sorted([f for f in results['overall_per_field_acc'].keys() if f != 'scam_incident_description'])
        field_labels = [clean_label(f) for f in fields]
        acc_values = [results['overall_per_field_acc'][f] for f in fields]
        
        fig, ax = plt.subplots(figsize=(16, 8))
        bars = ax.bar(field_labels, acc_values, color='skyblue')
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=20)
        ax.set_title('Accuracy: Per-Field (Overall)', fontsize=20)
        ax.set_ylabel('Accuracy', fontsize=18)
        ax.set_ylim(0, 1.1)
        ax.tick_params(axis='x', rotation=90, labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        plt.tight_layout()
        self.save_figure(fig, str(scam_details_dir / 'per_field_accuracy_bar.png'))

        # Confusion heatmaps for categorical fields
        for field, conf in results['confusion_per_field'].items():
            labels = sorted(set(conf.keys()) | {k for sub in conf.values() for k in sub.keys()})
            cleaned_labels = [clean_label(l) for l in labels]
            matrix = np.zeros((len(labels), len(labels)))
            label_to_idx = {label: i for i, label in enumerate(labels)}
                        
            for gt, pred_dict in conf.items():
                for pred, count in pred_dict.items():
                    matrix[label_to_idx[gt]][label_to_idx[pred]] = count
                        
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues', xticklabels=cleaned_labels, yticklabels=cleaned_labels, ax=ax,
                        annot_kws={'size': 20})
            ax.set_title(f'Confusion Matrix: {clean_label(field)}', fontsize=20)
            ax.set_xlabel('Predicted', fontsize=18)
            ax.set_ylabel('Actual', fontsize=18)
            ax.tick_params(axis='both', labelsize=18)
            plt.tight_layout()
            self.save_figure(fig, str(scam_details_dir / f'confusion_{field}.png'))

        # Heatmap for accuracies by model and permutation (structured)
        desired_order = [
            'high_high_neutral', 'high_high_distressed',
            'high_low_neutral', 'high_low_distressed',
            'low_high_neutral', 'low_high_distressed',
            'low_low_neutral', 'low_low_distressed'
        ]
        segment_map = {
            'high_high_neutral': 'HHN', 'high_high_distressed': 'HHD',
            'high_low_neutral': 'HLN', 'high_low_distressed': 'HLD',
            'low_high_neutral': 'LHN', 'low_high_distressed': 'LHD',
            'low_low_neutral': 'LLN', 'low_low_distressed': 'LLD'
        }
        models = sorted(results['by_police_model_and_permutation'].keys())
        perms = [s for s in desired_order if s in set(s for m in models for s in results['by_police_model_and_permutation'][m])]
        short_perms = [segment_map[s] for s in perms]
        data = np.array([[results['by_police_model_and_permutation'][m].get(p, np.nan) for p in perms] for m in models])
        
        fig, ax = plt.subplots(figsize=(15, 8))
        im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(short_perms)))
        ax.set_xticklabels(short_perms, fontsize=18)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=18)
        ax.set_title('Accuracy: Model vs Victim Permutation', fontsize=20)
        for i in range(len(models)):
            for j in range(len(perms)):
                if not np.isnan(data[i, j]):
                    text = f'{data[i, j]:.2f}'
                else:
                    text = '-'
                ax.text(j, i, text, ha='center', va='center', color='black', fontsize=20)
        plt.tight_layout()
        self.save_figure(fig, str(scam_details_dir / 'heatmap_model_vs_permutation.png'))

        # Histogram for description similarities
        if results['all_desc_similarities']:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.hist(results['all_desc_similarities'], bins=20, color='green')
            ax.set_title('Distribution of Description Semantic Similarities', fontsize=20)
            ax.set_xlabel('Cosine Similarity', fontsize=18)
            ax.set_ylabel('Frequency', fontsize=18)
            ax.tick_params(axis='both', labelsize=18)
            plt.tight_layout()
            self.save_figure(fig, str(scam_details_dir / 'desc_similarity_hist.png'))

        # Accuracy over sorted conversations 
        plot_data = pd.DataFrame([
            {'convo_id': int(c), 'accuracy': info['accuracy'], 'model': info['police_llm_model']}
            for c, info in results['per_conversation'].items()
        ])
        plot_data = plot_data.dropna(subset=['accuracy']).sort_values('convo_id')
        # Limit to first 48 conversations
        plot_data = plot_data.groupby('convo_id').head(48)
        if len(plot_data['convo_id'].unique()) >= 2:
            # Aggregate accuracies across models for each convo_id
            aggregated_data = plot_data.groupby('convo_id')['accuracy'].mean().reset_index()
            aggregated_data = aggregated_data.sort_values('convo_id').head(48) 
            if len(aggregated_data) >= 2:
                fig, ax = plt.subplots(figsize=(14, 7))
                x = list(range(1, len(aggregated_data) + 1))
                y = aggregated_data['accuracy'].values
                ax.plot(x, y, color='blue', alpha=0.7, linewidth=2, label='Average Accuracy (All Models)')
                X = np.array(x).reshape(-1, 1)
                reg = LinearRegression().fit(X, y)
                trend = reg.predict(X)
                ax.plot(x, trend, linestyle='--', color='red', linewidth=2, label='Trend')
                ax.set_title('Overall Accuracy Over First 48 Conversations (All Models)', fontsize=20)
                ax.set_xlabel('Conversation Index (Sorted by ID)', fontsize=18)
                ax.set_ylabel('Accuracy', fontsize=18)
                ax.set_ylim(0, 1)
                ax.set_xlim(1, 48)
                ax.tick_params(axis='both', labelsize=18)
                ax.legend(fontsize=18)
                ax.grid(True)
                plt.tight_layout()
                self.save_figure(fig, str(scam_details_dir / 'accuracy_over_convos_overall.png'))
            else:
                self.logger.warning("Insufficient data points after aggregation (< 2 conversations)")
        else:
            self.logger.warning("Insufficient unique conversations (< 2)")
    
        # Accuracy over sorted conversations (combined models)
        plot_data = pd.DataFrame([
            {'convo_id': int(c), 'accuracy': info['accuracy'], 'model': info['police_llm_model']}
            for c, info in results['per_conversation'].items()
        ])
        plot_data = plot_data.dropna(subset=['accuracy'])
        models = sorted(plot_data['model'].unique())
        color_list = list(mcolors.TABLEAU_COLORS.values())
        colors = {model: color_list[i % len(color_list)] for i, model in enumerate(models)}
        for model in models:
            model_lower = model.lower()
            if 'gpt-4o-mini' in model_lower:
                colors[model] = '#2CA02C'
            elif 'mistral' in model_lower:
                colors[model] = '#D62728'
            elif model_lower == 'qwen2.5:7b':
                colors[model] = '#FF7F0E'
            elif model_lower == 'granite3.2:8b':
                colors[model] = '#1F77B4'
        fig, ax = plt.subplots(figsize=(14, 7))
        for model in models:
            model_data = plot_data[plot_data['model'] == model].sort_values('convo_id')
            if len(model_data) < 2:
                self.logger.warning(f"Skipping model {model}: insufficient data points ({len(model_data)})")
                continue
            x = list(range(1, len(model_data) + 1))
            y = model_data['accuracy'].values
            ax.plot(x, y, label=f"{model} (Raw)", color=colors[model], linewidth=2, alpha=0.3)
            X = np.array(x).reshape(-1, 1)
            reg = LinearRegression().fit(X, y)
            trend = reg.predict(X)
            ax.plot(x, trend, linestyle='--', color=colors[model], linewidth=2, label=f"{model} (Trend)")
        ax.set_title('Accuracy Over Sorted Conversations by Model', fontsize=20)
        ax.set_xlabel('Conversation Index (Sorted by ID)', fontsize=18)
        ax.set_ylabel('Accuracy', fontsize=18)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='both', labelsize=18)
        ax.legend(fontsize=18, ncol=2)
        ax.grid(True)
        plt.tight_layout()
        self.save_figure(fig, str(scam_details_dir / 'accuracy_over_convos_combined.png'))

        # Average accuracy by turn position per model (no trend lines)
        models = sorted(avg_per_turn_by_model.keys())
        color_list = list(mcolors.TABLEAU_COLORS.values())
        colors = {model: color_list[i % len(color_list)] for i, model in enumerate(models)}
        for model in models:
            model_lower = model.lower()
            if 'gpt-4o-mini' in model_lower:
                colors[model] = '#2CA02C'
            elif 'mistral' in model_lower:
                colors[model] = '#D62728'
            elif model_lower == 'qwen2.5:7b':
                colors[model] = '#FF7F0E'
            elif model_lower == 'granite3.2:8b':
                colors[model] = '#1F77B4'
        max_length = max_turn + 1
        fig, ax = plt.subplots(figsize=(12, 6))
        for model in models:
            x = range(1, max_length + 1)
            y = avg_per_turn_by_model[model]
            ax.plot(x, y, label=model, color=colors[model], linewidth=3)
        ax.set_title('Average Accuracy by Turn Position (Per Model)', fontsize=20)
        ax.set_xlabel('Turn Position', fontsize=18)
        ax.set_ylabel('Accuracy', fontsize=18)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='both', labelsize=18)
        ax.legend(fontsize=18, ncol=2)
        ax.grid(True)
        overall_count_per_turn = [sum(count_per_turn_by_model[model][turn] for model in models) for turn in range(max_turn + 1)]
        ax2 = ax.twinx()
        ax2.plot(range(1, max_length + 1), overall_count_per_turn, color='gray', linestyle='--', label='Total Convo Count', linewidth=2)
        ax2.set_ylabel('Number of Conversations', fontsize=18)
        ax2.tick_params(axis='y', labelsize=18)
        ax2.legend(loc='lower left', fontsize=18)
        plt.tight_layout()
        self.save_figure(fig, str(scam_details_dir / 'line_avg_by_turn_position_per_model.png'))

        scam_types = sorted(set(results['confusion_per_field']['scam_type'].keys()))  # GT scam types
        cleaned_scam_types = [clean_label(t) for t in scam_types]
        data = np.full((len(scam_types), len(models)), np.nan) 
        for i, scam_type in enumerate(scam_types):
            for j, model in enumerate(models):
                data[i, j] = results['by_model_scam_type_acc'].get(model, {}).get(scam_type, np.nan)

        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, fontsize=18, rotation=0, ha='center')
        ax.set_yticks(range(len(cleaned_scam_types)))
        ax.set_yticklabels(cleaned_scam_types, fontsize=18)
        ax.set_title('Overall Accuracy: Model vs Ground Truth Scam Type', fontsize=20)
        for i in range(len(scam_types)):
            for j in range(len(models)):
                if not np.isnan(data[i, j]):
                    text = f'{data[i, j]:.2f}'
                else:
                    text = '-'
                ax.text(j, i, text, ha='center', va='center', color='black', fontsize=20)
        plt.tight_layout()
        self.save_figure(fig, str(scam_details_dir / 'heatmap_model_vs_scam_type.png'))

        # Heatmap for model vs fields
        fields = sorted([f for f in results['overall_per_field_acc'].keys() if f != 'scam_incident_description'])
        field_labels = [clean_label(f) for f in fields]
        data = np.array([[results['by_model_per_field_acc'][m].get(f, np.nan) for f in fields] for m in models])
        
        fig, ax = plt.subplots(figsize=(15, 8))
        im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(field_labels)))
        ax.set_xticklabels(field_labels, fontsize=18, rotation=90, ha='center')
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=18)
        ax.set_title('Accuracy: Model vs Field', fontsize=20)
        for i in range(len(models)):
            for j in range(len(fields)):
                if not np.isnan(data[i, j]):
                    text = f'{data[i, j]:.2f}'
                else:
                    text = '-'
                ax.text(j, i, text, ha='center', va='center', color='black', fontsize=20)
        plt.tight_layout()
        self.save_figure(fig, str(scam_details_dir / 'heatmap_model_vs_field.png'))

    def analyze_mismatches(self, results):
        """
        Print detailed mismatch analysis.
        """
        if not results['all_mismatches']:
            self.logger.info("No mismatches for analysis.")
            return
        
        mismatch_counts = Counter([m[3] for m in results['all_mismatches']])
        self.logger.info("Mismatch Counts per Field:")
        for field, count in mismatch_counts.items():
            self.logger.info(f"{field}: {count}")
        
        mismatch_by_model_field = defaultdict(Counter)
        for m in results['all_mismatches']:
            mismatch_by_model_field[m[1]][m[3]] += 1
        self.logger.info("\nMismatches by Model and Field:")
        for model, field_count in mismatch_by_model_field.items():
            self.logger.info(f"Model {model}: {dict(field_count)}")
        
        mismatch_by_perm_field = defaultdict(Counter)
        for m in results['all_mismatches']:
            mismatch_by_perm_field[m[2]][m[3]] += 1
        self.logger.info("\nMismatches by Permutation and Field:")
        for perm, field_count in mismatch_by_perm_field.items():
            self.logger.info(f"Permutation {perm}: {dict(field_count)}")
        
        self.logger.info("\nSample Mismatches (first 5 per field):")
        mismatches_by_field = defaultdict(list)
        for m in results['all_mismatches']:
            mismatches_by_field[m[3]].append(m)
        for field, ms in mismatches_by_field.items():
            self.logger.info(f"{field}:")
            for sample in ms[:5]:
                convo, model, perm, _, gt_raw, gt_norm, pred_raw, pred_norm = sample
                self.logger.info(f" Convo {convo} (Model {model}, Perm {perm}): GT {gt_raw} ({gt_norm}), Pred {pred_raw} ({pred_norm})")

    def compute_advanced_metrics(self, results):
        """
        Compute precision, recall, F1, and description similarity stats.
        """
        for field, conf in results['confusion_per_field'].items():
            y_true = []
            y_pred = []
            for gt, pred_dict in conf.items():
                for pred, count in pred_dict.items():
                    y_true.extend([gt] * count)
                    y_pred.extend([pred] * count)
            
            if not y_true:
                continue
            
            prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            
            self.logger.info(f"\nAdvanced Metrics for {field}:")
            self.logger.info(f"Precision (macro): {prec:.2f}")
            self.logger.info(f"Recall (macro): {rec:.2f}")
            self.logger.info(f"F1 (macro): {f1:.2f}")
        
        if results['all_desc_similarities']:
            mean_sim = np.mean(results['all_desc_similarities'])
            std_sim = np.std(results['all_desc_similarities'])
            self.logger.info("\nDescription Similarity Stats:")
            self.logger.info(f"Mean: {mean_sim:.2f}")
            self.logger.info(f"Std Dev: {std_sim:.2f}")

    def perform_eda(self, last_rows):
        """
        Perform basic EDA.
        """
        unique_models = last_rows['police_llm_model'].unique().tolist()
        profile_ids = last_rows['profile_id'].unique()
        permutations = []
        for pid in profile_ids:
            user_prof = self.ground_truth_profile.get(pid, {})
            tl = self.normalize_profile_value(user_prof.get('tech_literacy', 'na'))
            lp = self.normalize_profile_value(user_prof.get('language_proficiency', 'na'))
            es = self.normalize_profile_value(user_prof.get('emotional_state', 'na'))
            perm = f"{tl}_{lp}_{es}"
            permutations.append(perm)
        unique_permutations = sorted(list(set(permutations)))
        
        eda_results = {
            'unique_models': unique_models,
            'unique_permutations': unique_permutations,
            'num_conversations': len(last_rows)
        }
        self.logger.info(f"EDA: {eda_results}")
        return eda_results

    def run(self) -> dict:
        """
        Run the full calculation.
        """
        self.load_data()
        last_rows = self.preprocess_csv()
        eda_results = self.perform_eda(last_rows)
        accuracies = self.compute_accuracies(last_rows)
        avg_per_turn_by_model, count_per_turn_by_model, max_turn = self.compute_per_turn_accuracies()
        self.generate_charts(accuracies, accuracies['per_conversation'], avg_per_turn_by_model, count_per_turn_by_model, max_turn)
        self.analyze_mismatches(accuracies)
        self.compute_advanced_metrics(accuracies)
        return {**accuracies, 'eda': eda_results}

if __name__ == "__main__":
    csv_path = "simulations/final_results/phase_2/profile_rag_ie_kb/autonomous_conversation_history.csv"
    json_path = "data/victim_profile/victim_details.json"
    
    calculator = ScamAccuracyCalculator(csv_path, json_path)
    results = calculator.run()
    
    print("Per Conversation Accuracies and Description Similarities:")
    low_sim_convos = []
    for convo_id, info in sorted(results['per_conversation'].items()):
        if info['desc_similarity'] < 0.7:
            low_sim_convos.append((convo_id, info['profile_id'], info['police_llm_model'], info['desc_similarity']))
    
    print(f"\nOverall Structured Accuracy: {results['overall_accuracy']:.2f}")
    print(f"Overall Description Similarity: {results['overall_desc_similarity']:.2f}")
    
    if results['all_desc_similarities']:
        mean_sim = np.mean(results['all_desc_similarities'])
        std_sim = np.std(results['all_desc_similarities'])
        print(f"Description Similarity Stats: Mean: {mean_sim:.2f}, Std Dev: {std_sim:.2f}")
        calculator.logger.info(f"Description Similarity Stats: Mean: {mean_sim:.2f}, Std Dev: {std_sim:.2f}")
        
        bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist, _ = np.histogram(results['all_desc_similarities'], bins=bins)
        total = len(results['all_desc_similarities'])
        dist_summary = "\nDescription Similarity Distribution:"
        for i in range(len(hist)):
            percentage = (hist[i] / total * 100) if total > 0 else 0
            dist_summary += f"\n {bins[i]:.1f}–{bins[i+1]:.1f}: {percentage:.1f}% ({hist[i]} convos)"
        print(dist_summary)
        calculator.logger.info(dist_summary)
    
    if low_sim_convos:
        print("\nConversations with Low Description Similarity (<0.7):")
        for convo_id, profile_id, model, sim in low_sim_convos:
            print(f" Convo {convo_id} (Profile {profile_id}, Model {model}): Desc Sim: {sim:.2f}")
        calculator.logger.info("\nConversations with Low Description Similarity (<0.7):")
        for convo_id, profile_id, model, sim in low_sim_convos:
            calculator.logger.info(f" Convo {convo_id} (Profile {profile_id}, Model {model}): Desc Sim: {sim:.2f}")
    
    print("\nAccuracy and Desc Similarity by Police LLM Model:")
    for model, acc in sorted(results['by_police_model'].items()):
        print(f"{model}: Structured Acc: {acc:.2f}, Desc Sim: {results['by_model_desc_similarity'][model]:.2f}")
    
    print("\nAccuracy and Desc Similarity by Victim Permutation (GT):")
    for perm, acc in sorted(results['by_victim_permutation'].items()):
        print(f"{perm}: Structured Acc: {acc:.2f}, Desc Sim: {results['by_permutation_desc_similarity'][perm]:.2f}")
    
    print("\nAccuracy and Desc Similarity by Police LLM Model and Victim Permutation (GT):")
    for model, perm_accs in sorted(results['by_police_model_and_permutation'].items()):
        print(f"Model: {model}")
        for perm, acc in sorted(perm_accs.items()):
            print(f" - {perm}: Structured Acc: {acc:.2f}, Desc Sim: {results['by_model_permutation_desc_similarity'][model].get(perm, 0.0):.2f}")
    
    print("\nOverall Per-Field Structured Accuracy:")
    for attr, acc in results['overall_per_field_acc'].items():
        print(f"{attr}: {acc:.2f}")
    
    print("\nPer-Field Structured Accuracy by Model:")
    for model, attr_accs in results['by_model_per_field_acc'].items():
        print(f"Model: {model}")
        for attr, acc in attr_accs.items():
            print(f" - {attr}: {acc:.2f}")
    
    print("\nConfusion Matrices for Structured Fields:")
    for attr, conf in results['confusion_per_field'].items():
        print(f"{attr}:")
        for gt, pred_dict in conf.items():
            print(f" Actual {gt}: {dict(pred_dict)}")