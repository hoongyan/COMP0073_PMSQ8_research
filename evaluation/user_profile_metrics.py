import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.logging_config import setup_logger
from config.settings import get_settings

import pandas as pd
import json
import ast

import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import numpy as np
from statistics import mean, median
from collections import defaultdict, Counter

class ProfileAccuracyCalculator:
    def __init__(self, csv_path: str, json_path: str):
        """
        Initialize with file paths.
        """
        self.csv_path = csv_path
        self.json_path = json_path
        self.df = None  
        self.ground_truth_profile = None  
        self.settings = get_settings()
        self.logger = setup_logger("user_profile", self.settings.log.subdirectories["evaluation"])

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
            self.ground_truth_profile = {item['profile_id']: item['user_profile'] for item in ground_truth_list}
            self.logger.info(f"Loaded JSON with {len(ground_truth_list)} entries.")
        except FileNotFoundError:
            self.logger.error(f"JSON file not found: {self.json_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading JSON: {e}")
            raise

    def save_figure(self, fig, filepath: str) -> None:
        """
        Saves a Matplotlib figure to the given filepath.
        Assumes the parent dir already exists (created once upstream).
        Closes the figure to free memory.
        
        Args:
            fig: The Matplotlib figure to save (use plt.gcf() for current).
            filepath: Full path to save (str or Path).
        """
        try:
            fig.savefig(filepath)
            self.logger.info(f"Saved figure to '{filepath}'")
        except Exception as e:
            self.logger.error(f"Failed to save figure to '{filepath}': {e}")
            raise
        finally:
            plt.close(fig)
            
    def preprocess_csv(self):
        """
        Preprocess CSV: Convert timestamps, filter for police rows, group by conversation_id, take first police row (min timestamp) for initial_profile.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], errors='coerce')
        
        # Filter for police rows only
        police_df = self.df[self.df['sender_type'] == 'police']
        
        if police_df.empty:
            raise ValueError("No police rows found in CSV. Cannot compute accuracies.")
        
        # Group by conversation_id
        first_police_rows = police_df.loc[police_df.groupby('conversation_id')['timestamp'].idxmin()].set_index('conversation_id')
        
        self.logger.info(f"Preprocessed CSV to {len(first_police_rows)} unique conversations using first police row per convo.")
        
        all_convos = set(self.df['conversation_id'].unique())
        processed_convos = set(first_police_rows.index)
        missing = all_convos - processed_convos
        if missing:
            self.logger.warning(f"Conversations without police rows (skipped): {missing}")
        
        return first_police_rows

    def parse_profile(self, pred_str):
        """
        Parse initial_profile string to dict with 'level' and 'confidence' for each key.
        Expected format: {'tech_literacy': {'score': float, 'level': str, 'confidence': float}, ...}
        """
        if pd.isna(pred_str) or not pred_str.strip():
            return {}
        try:
            raw_dict = ast.literal_eval(pred_str)
            # Extract 'level' and 'confidence'
            return {
                k: {
                    'level': v.get('level', 'na') if isinstance(v, dict) else 'na',
                    'confidence': v.get('confidence', 0.0) if isinstance(v, dict) else 0.0
                } for k, v in raw_dict.items()
            }
        except (ValueError, SyntaxError) as e:
            self.logger.warning(f"Error parsing initial_profile: {e}. Returning empty dict.")
            return {}

    def normalize_value(self, key, val):
        """
        Normalize profile values: lowercase strings, handle nulls.
        Since profile fields are categorical ('low'/'high', 'distressed'/'neutral').
        """
        if val is None or val == 'na':
            return 'na'
        val = str(val).strip().lower()
        null_synonyms = {'', 'na', 'n/a', 'none', 'not applicable', 'not available', 'not provided', 'unknown'}
        if val in null_synonyms:
            return 'na'
        return val

    def compare_profiles(self, pred_dict, gt_dict):
        """
        Compare normalized predicted profile levels to ground truth.
        Keys: tech_literacy, language_proficiency, emotional_state.
        Returns correct count, total count (always 3 if all present in GT).
        Also returns per-attribute correct (dict), and mismatches (list of tuples: (key, gt_val, pred_val, confidence))
        """
        keys_to_compare = ['tech_literacy', 'language_proficiency', 'emotional_state']
        correct = 0
        total = len(keys_to_compare)
        per_attr_correct = {key: 0 for key in keys_to_compare} 
        mismatches = []  
        
        for key in keys_to_compare:
            if key not in gt_dict:
                total -= 1  
                continue
            if key not in pred_dict:
                mismatches.append((key, gt_dict[key], 'missing', 0.0))  
                continue
            
            norm_pred = self.normalize_value(key, pred_dict[key]['level'])
            norm_gt = self.normalize_value(key, gt_dict[key])
            conf = pred_dict[key]['confidence']
            
            if norm_pred == norm_gt:
                correct += 1
                per_attr_correct[key] = 1  # Correct for this attr
            else:
                mismatches.append((key, norm_gt, norm_pred, conf))  # Record mismatch with conf
        
        return correct, total, per_attr_correct, mismatches

    def compute_accuracies(self, first_rows):
        """
        Compute all accuracies: per convo, overall, by model, by permutation, and by model+permutation.
        Add per-attribute accuracies (overall, by model, by permutation, by model+permutation).
        Confusion matrices per attribute (dict of counters: actual -> predicted -> count).
        Collect all mismatches with convo_id for detailed analysis.
        Collect all_comparisons for confidence analysis (list of dicts: convo_id, model, permutation, attr, gt, pred, conf, is_correct).
        """
        per_convo_accuracies = {}
        overall_correct = 0
        overall_total = 0
        
        by_police_model = {}  

        by_permutation = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        by_police_model_and_permutation = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
        
        overall_per_attr_correct = defaultdict(int)  
        overall_per_attr_total = defaultdict(int)  
        by_model_per_attr = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))  
        by_permutation_per_attr = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))  
        by_model_permutation_per_attr = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0})))  
        
        # Confusion matrices 
        confusion_per_attr = {key: defaultdict(lambda: defaultdict(int)) for key in ['tech_literacy', 'language_proficiency', 'emotional_state']}
        
        # Confusion matrices per model and attr
        confusion_per_attr_by_model = defaultdict(lambda: {key: defaultdict(lambda: defaultdict(int)) for key in ['tech_literacy', 'language_proficiency', 'emotional_state']})
        
        # All mismatches
        all_mismatches = []
        
        # All comparisons for confidence analysis
        all_comparisons = []
        
        for convo_id, row in first_rows.iterrows():
            profile_id = row['profile_id']
            if profile_id not in self.ground_truth_profile:
                self.logger.warning(f"Profile ID {profile_id} for convo {convo_id} not in ground truth. Skipping.")
                continue
            
            gt_dict = self.ground_truth_profile[profile_id]
            pred_dict = self.parse_profile(row['initial_profile'])
            
            correct, total, per_attr_correct, mismatches = self.compare_profiles(pred_dict, gt_dict)
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
            
            # Group by police_llm_model
            model = row['police_llm_model']
            if model not in by_police_model:
                by_police_model[model] = {'correct': 0, 'total': 0}
            by_police_model[model]['correct'] += correct
            by_police_model[model]['total'] += total
            
            # Group by permutation (based on GT)
            user_prof = gt_dict
            tech = self.normalize_value('tech_literacy', user_prof.get('tech_literacy'))
            lang = self.normalize_value('language_proficiency', user_prof.get('language_proficiency'))
            emot = self.normalize_value('emotional_state', user_prof.get('emotional_state'))
            seg = f"{tech}_{lang}_{emot}"
            
            by_permutation[seg]['correct'] += correct
            by_permutation[seg]['total'] += total
            
            # Group by model and permutation
            if model not in by_police_model_and_permutation:
                by_police_model_and_permutation[model] = {
                    'low_low_distressed': {'correct': 0, 'total': 0},
                    'low_low_neutral': {'correct': 0, 'total': 0},
                    'low_high_distressed': {'correct': 0, 'total': 0},
                    'low_high_neutral': {'correct': 0, 'total': 0},
                    'high_low_distressed': {'correct': 0, 'total': 0},
                    'high_low_neutral': {'correct': 0, 'total': 0},
                    'high_high_distressed': {'correct': 0, 'total': 0},
                    'high_high_neutral': {'correct': 0, 'total': 0},
                    'other': {'correct': 0, 'total': 0}
                }
            by_police_model_and_permutation[model][seg]['correct'] += correct
            by_police_model_and_permutation[model][seg]['total'] += total
            
            # Per-attribute updates and all_comparisons
            for attr in ['tech_literacy', 'language_proficiency', 'emotional_state']:
                if attr not in gt_dict:
                    continue
                norm_gt = self.normalize_value(attr, gt_dict[attr])
                attr_total = 1
                if attr not in pred_dict:
                    norm_pred = 'missing'
                    conf = 0.0
                    attr_correct = 0
                else:
                    norm_pred = self.normalize_value(attr, pred_dict[attr]['level'])
                    conf = pred_dict[attr]['confidence']
                    attr_correct = 1 if norm_pred == norm_gt else 0
                
                overall_per_attr_correct[attr] += attr_correct
                overall_per_attr_total[attr] += attr_total
                
                # By model
                by_model_per_attr[model][attr]['correct'] += attr_correct
                by_model_per_attr[model][attr]['total'] += attr_total
                
                # By permutation
                by_permutation_per_attr[seg][attr]['correct'] += attr_correct
                by_permutation_per_attr[seg][attr]['total'] += attr_total
                
                # By model + permutation
                by_model_permutation_per_attr[model][seg][attr]['correct'] += attr_correct
                by_model_permutation_per_attr[model][seg][attr]['total'] += attr_total
                
                # Confusion: Add to matrix
                confusion_per_attr[attr][norm_gt][norm_pred] += 1
                
                confusion_per_attr_by_model[model][attr][norm_gt][norm_pred] += 1
                
                # Add to all_comparisons
                all_comparisons.append({
                    'convo_id': convo_id,
                    'model': model,
                    'permutation': seg,
                    'attr': attr,
                    'gt': norm_gt,
                    'pred': norm_pred,
                    'conf': conf,
                    'is_correct': (attr_correct == 1)
                })
            
            for mismatch in mismatches:
                all_mismatches.append((convo_id, *mismatch))
        
        # Compute overall accuracy
        overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0.0
        
        # Compute by_model
        by_police_model_acc = {model: data['correct'] / data['total'] if data['total'] > 0 else 0.0 
                               for model, data in by_police_model.items()}
        
        # Compute by_permutation
        by_permutation_acc = {seg: data['correct'] / data['total'] if data['total'] > 0 else 0.0 
                                 for seg, data in by_permutation.items()}
        
        # Compute by_model_and_permutation
        by_police_model_and_permutation_acc = {}
        for model, seg_dict in by_police_model_and_permutation.items():
            by_police_model_and_permutation_acc[model] = {}
            for seg, data in seg_dict.items():
                by_police_model_and_permutation_acc[model][seg] = data['correct'] / data['total'] if data['total'] > 0 else 0.0
        
        # Compute per-attribute accuracies
        overall_per_attr_acc = {attr: overall_per_attr_correct[attr] / overall_per_attr_total[attr] if overall_per_attr_total[attr] > 0 else 0.0
                                for attr in overall_per_attr_total}
        
        by_model_per_attr_acc = {}
        for model, attr_dict in by_model_per_attr.items():
            by_model_per_attr_acc[model] = {attr: data['correct'] / data['total'] if data['total'] > 0 else 0.0
                                            for attr, data in attr_dict.items()}
        
        by_permutation_per_attr_acc = {}
        for seg, attr_dict in by_permutation_per_attr.items():
            by_permutation_per_attr_acc[seg] = {attr: data['correct'] / data['total'] if data['total'] > 0 else 0.0
                                            for attr, data in attr_dict.items()}
        
        by_model_permutation_per_attr_acc = {}
        for model, seg_dict in by_model_permutation_per_attr.items():
            by_model_permutation_per_attr_acc[model] = {}
            for seg, attr_dict in seg_dict.items():
                by_model_permutation_per_attr_acc[model][seg] = {attr: data['correct'] / data['total'] if data['total'] > 0 else 0.0
                                                             for attr, data in attr_dict.items()}
        
        return {
            'per_conversation': per_convo_accuracies,
            'overall_accuracy': overall_accuracy,
            'by_police_model': by_police_model_acc,
            'by_permutation': by_permutation_acc,
            'by_police_model_and_permutation': by_police_model_and_permutation_acc,
            'overall_per_attr_acc': overall_per_attr_acc,
            'by_model_per_attr_acc': by_model_per_attr_acc,
            'by_permutation_per_attr_acc': by_permutation_per_attr_acc,
            'by_model_permutation_per_attr_acc': by_model_permutation_per_attr_acc,
            'confusion_per_attr': confusion_per_attr,
            'all_mismatches': all_mismatches,
            'all_comparisons': all_comparisons, 
            'confusion_per_attr_by_model': confusion_per_attr_by_model,
        }

    def perform_eda(self, first_rows):
        """
        Exploratory Data Analysis.
        - Distributions of GT and predicted labels per attribute.
        - NEW: Distribution of confidences (histogram data), counts per model/permutation.
        Returns dict of counters.
        """
        gt_dist = {attr: Counter() for attr in ['tech_literacy', 'language_proficiency', 'emotional_state']}
        pred_dist = {attr: Counter() for attr in ['tech_literacy', 'language_proficiency', 'emotional_state']}
        confidences = []  
        
        model_counts = Counter()
        permutation_counts = Counter()
        profile_counts = Counter()
        
        for convo_id, row in first_rows.iterrows():
            profile_id = row['profile_id']
            if profile_id not in self.ground_truth_profile:
                continue
            gt_dict = self.ground_truth_profile[profile_id]
            pred_dict = self.parse_profile(row['initial_profile'])
            
            model = row['police_llm_model']
            model_counts[model] += 1
            profile_counts[profile_id] += 1
            
            user_prof = gt_dict
            tech = self.normalize_value('tech_literacy', user_prof.get('tech_literacy'))
            lang = self.normalize_value('language_proficiency', user_prof.get('language_proficiency'))
            emot = self.normalize_value('emotional_state', user_prof.get('emotional_state'))
            seg = f"{tech}_{lang}_{emot}"
            permutation_counts[seg] += 1
            
            for attr in gt_dist:
                if attr in gt_dict:
                    norm_gt = self.normalize_value(attr, gt_dict[attr])
                    gt_dist[attr][norm_gt] += 1
                if attr in pred_dict:
                    norm_pred = self.normalize_value(attr, pred_dict[attr]['level'])
                    pred_dist[attr][norm_pred] += 1
                    confidences.append(pred_dict[attr]['confidence'])
        
        self.logger.info("EDA Distributions:")
        for attr in gt_dist:
            self.logger.info(f"GT {attr}: {dict(gt_dist[attr])}")
            self.logger.info(f"Pred {attr}: {dict(pred_dist[attr])}")
        
        self.logger.info(f"Model Counts: {dict(model_counts)}")
        self.logger.info(f"Permutation Counts: {dict(permutation_counts)}")
        self.logger.info(f"Profile ID Counts: {dict(profile_counts)}")
        
        return {'gt_dist': gt_dist, 'pred_dist': pred_dist, 'confidences': confidences, 'model_counts': model_counts, 'permutation_counts': permutation_counts}

    def generate_charts(self, results):
        """
        Generate visualizations.
        - Bar chart for overall and per-attribute accuracies.
        - Heatmap for confusion matrices (one per attribute).
        - Boxplot for confidences by attribute and correctness (using matplotlib).
        - Stacked bar for accuracies by model and permutation.
        - Bar chart for accuracies by permutation.
        - Heatmap for accuracies by model and permutation.
        Saves to files.
        """
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
        'high_high_neutral',
        'high_high_distressed',
        'high_low_neutral',
        'high_low_distressed',
        'low_high_neutral',
        'low_high_distressed',
        'low_low_neutral',
        'low_low_distressed'
    ]
        
        base_eval_dir = Path(self.settings.data.evaluation_results_dir)
        user_profile_dir = base_eval_dir / 'user_profile'
        user_profile_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Ensured output directory exists: {user_profile_dir}")
        
        plot_paths = {
            'accuracy_bar': str(user_profile_dir / 'accuracy_bar.png'),
            'conf_boxplot': str(user_profile_dir / 'conf_boxplot.png'),
            'accuracy_per_permutation': str(user_profile_dir / 'accuracy_per_permutation.png'),
            'heatmap_model_permutation': str(user_profile_dir / 'heatmap_model_permutation.png'),
            'heatmap_model_attr': str(user_profile_dir / 'heatmap_model_attr.png'), 
            'heatmap_attr_perm': str(user_profile_dir / 'heatmap_attr_perm.png'),
        }
        
        # Bar chart for accuracies
        attrs = ['overall'] + list(results['overall_per_attr_acc'].keys())
        acc_values = [results['overall_accuracy']] + list(results['overall_per_attr_acc'].values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(attrs, acc_values, color='skyblue', width=0.4)
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=16)
            
        plt.title('Accuracy: Overall and Per-Attribute', fontsize=20)

        plt.ylabel('Accuracy', fontsize=16)
        plt.xlabel('Attributes', fontsize=16) 
        plt.xticks(fontsize=16, rotation=0)  
        plt.yticks(fontsize=16)
        plt.ylim(0, 1.1)  
        self.save_figure(plt.gcf(), plot_paths['accuracy_bar'])
        self.logger.info("Saved accuracy bar chart to 'accuracy_bar.png'")
        
        attrs = ['tech_literacy', 'language_proficiency', 'emotional_state']

        fig, axs = plt.subplots(1, 3, figsize=(24, 8))
        for i, attr in enumerate(attrs):
            conf = results['confusion_per_attr'][attr]
            labels = sorted(set(list(conf.keys()) + [k for sub in conf.values() for k in sub.keys()]))  # Unique labels
            matrix = np.zeros((len(labels), len(labels)))
            label_to_idx = {label: i for i, label in enumerate(labels)}
            
            for gt, pred_dict in conf.items():
                for pred, count in pred_dict.items():
                    matrix[label_to_idx[gt]][label_to_idx[pred]] = count
            
            im = axs[i].imshow(matrix, cmap='Blues')
            cbar=fig.colorbar(im, ax=axs[i], shrink=0.8)  
            cbar.ax.tick_params(labelsize=18)
            axs[i].set_xticks(range(len(labels)))
            axs[i].set_xticklabels(labels, rotation=0, fontsize=22)
            axs[i].set_yticks(range(len(labels)))
            axs[i].set_yticklabels(labels, fontsize=22)
            axs[i].set_xlabel('Predicted', fontsize=22)
            axs[i].set_ylabel('Actual', fontsize=22)
            axs[i].set_title(f'{attr}', fontsize=25)
            for row in range(len(labels)):
                for col in range(len(labels)):
                    value = matrix[row, col]
                    color = 'white' if value > matrix.max() / 2 else 'black'  
                    axs[i].text(col, row, int(value), ha='center', va='center', fontsize=24, color=color)  

        fig.suptitle('Overall Confusion Matrices by Profile Indicator', fontsize=25)
        plt.tight_layout()
        self.save_figure(fig, str(user_profile_dir / 'confusion_overall.png'))
        self.logger.info("Saved overall confusion subplots to 'confusion_overall.png'")

        # Model-specific confusion matrices 
        models = sorted(results['by_police_model'].keys())  
        num_models = len(models)
        if num_models > 0:
            fig, axs = plt.subplots(num_models, 3, figsize=(24, 8 * num_models), squeeze=False)  
            for row_idx, model in enumerate(models):
                for col_idx, attr in enumerate(attrs):
                    conf = results['confusion_per_attr_by_model'][model][attr]
                    labels = sorted(set(list(conf.keys()) + [k for sub in conf.values() for k in sub.keys()]))  
                    matrix = np.zeros((len(labels), len(labels)))
                    label_to_idx = {label: i for i, label in enumerate(labels)}
                    
                    for gt, pred_dict in conf.items():
                        for pred, count in pred_dict.items():
                            matrix[label_to_idx[gt]][label_to_idx[pred]] = count
                    
                    ax = axs[row_idx, col_idx]
                    im = ax.imshow(matrix, cmap='Blues')
                    fig.colorbar(im, ax=ax, shrink=0.8)
                    ax.set_xticks(range(len(labels)))
                    ax.set_xticklabels(labels, rotation=0, fontsize=20)  
                    ax.set_yticks(range(len(labels)))
                    ax.set_yticklabels(labels, fontsize=20)
                    ax.set_xlabel('Predicted', fontsize=20)
                    ax.set_ylabel('Actual', fontsize=20)
                    ax.set_title(f'{model} - {attr}', fontsize=20)

                    for r in range(len(labels)):
                        for c in range(len(labels)):
                            value = matrix[r, c]
                            color = 'white' if value > matrix.max() / 2 else 'black' 
                            ax.text(c, r, int(value), ha='center', va='center', fontsize=25, color=color)  
                
                axs[row_idx, 0].text(-0.5, 0.5, model, va='center', ha='right', fontsize=18, transform=axs[row_idx, 0].transAxes)

            fig.suptitle('Confusion Matrices by Model and Profile Indicator', fontsize=22)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
            self.save_figure(fig, str(user_profile_dir / 'confusion_by_model.png'))
            self.logger.info("Saved model-specific confusion subplots to 'confusion_by_model.png'")

        all_possible_perms = set(results['by_permutation'].keys())
        perms = [p for p in desired_order if p in all_possible_perms]
        accs = [results['by_permutation'][p] for p in perms]
        short_perms = [segment_map[p.replace('_', ' ')] for p in perms]
        plt.figure(figsize=(12, 8))
        bars = plt.bar(short_perms, accs, color='skyblue', width=0.4)
        plt.title('Accuracy by Profile Permutation', fontsize=20)
        plt.ylabel('Accuracy', fontsize=18)
        plt.xticks(rotation=0, fontsize=18)
        plt.yticks(fontsize=18)
        plt.ylim(0, 1.1)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=18)
        self.save_figure(plt.gcf(), plot_paths['accuracy_per_permutation'])
        self.logger.info("Saved accuracy by permutation to 'accuracy_per_permutation.png'")

        # Heatmap for accuracy by model and permutation
        models = sorted(results['by_police_model_and_permutation'].keys())
        all_possible_perms = set(results['by_permutation'].keys())
        perms = [p for p in desired_order if p in all_possible_perms]
        short_perms = [segment_map[p.replace('_', ' ')] for p in perms]
        data = np.array([[results['by_police_model_and_permutation'][m].get(p, 0.0) for p in perms] for m in models])
        plt.figure(figsize=(15, 6))
        plt.imshow(data, cmap='RdYlGn', vmin=0, vmax=1)
        cbar = plt.colorbar() 
        cbar.ax.tick_params(labelsize=18)
        plt.xticks(range(len(perms)), short_perms, rotation=0, fontsize=18)
        plt.yticks(range(len(models)), models, fontsize=18)
        plt.title('Accuracy Heatmap: Model vs Permutation',fontsize=20)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                plt.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', color='black', fontsize=18)
        self.save_figure(plt.gcf(), plot_paths['heatmap_model_permutation'])
        self.logger.info("Saved heatmap model vs permutation to 'heatmap_model_permutation.png'")

        # Heatmap for accuracy by model and attribute 
        models = sorted(results['by_police_model_and_permutation'].keys())
        attrs = ['tech_literacy', 'language_proficiency', 'emotional_state']
        data = np.array([[results['by_model_per_attr_acc'][m].get(a, 0.0) for a in attrs] for m in models]).T
        plt.figure(figsize=(15, 6))
        plt.imshow(data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=18)
        plt.xticks(range(len(models)), models, rotation=0, fontsize=18)  
        plt.yticks(range(len(attrs)), attrs, fontsize=18) 
        plt.title('Accuracy Heatmap: Profile Indicator vs Model', fontsize=20)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                plt.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', color='black', fontsize=18)
        plt.subplots_adjust(left=0.25)
        self.save_figure(plt.gcf(), plot_paths['heatmap_model_attr'])
        self.logger.info("Saved heatmap model vs attribute to 'heatmap_model_attr.png'")

        # Heatmap for accuracy by attribute (profile indicator) and victim segment (permutation)
        all_possible_perms = set(results['by_permutation'].keys())
        perms = [p for p in desired_order if p in all_possible_perms]
        short_perms = [segment_map[p.replace('_', ' ')] for p in perms]
        data = np.array([[results['by_permutation_per_attr_acc'][p].get(a, 0.0) for a in attrs] for p in perms]).T
        plt.figure(figsize=(15, 6))
        plt.imshow(data, cmap='RdYlGn', vmin=0, vmax=1)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=18)
        plt.xticks(range(len(short_perms)), short_perms, rotation=0, fontsize=18) 
        plt.yticks(range(len(attrs)), attrs, fontsize=18)  
        plt.title('Accuracy Heatmap: Victim Segment vs Profile Indicator', fontsize=20)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                plt.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', color='black', fontsize=18)
        plt.subplots_adjust(left=0.25)
        self.save_figure(plt.gcf(), plot_paths['heatmap_attr_perm'])
        self.logger.info("Saved heatmap attribute vs permutation to 'heatmap_attr_perm.png'")

    def analyze_mismatches(self, results):
        """
        Print detailed mismatch analysis.
        - Count mismatches per attribute.
        - Sample examples.
        """
        mismatch_counts = Counter([m[1] for m in results['all_mismatches']])  
        self.logger.info("Mismatch Counts per Attribute:")
        for attr, count in mismatch_counts.items():
            self.logger.info(f"{attr}: {count}")
        
        self.logger.info("\nSample Mismatches (first 5):")
        for mismatch in results['all_mismatches'][:5]:
            convo_id, attr, gt, pred, conf = mismatch
            self.logger.info(f"Convo {convo_id}: {attr} - GT: {gt}, Pred: {pred}, Conf: {conf:.2f}")

    def analyze_confidences(self, results):
        """
        Analyze confidence levels.
        - Create DF from all_comparisons.
        - Compute avg/median conf overall, by correct/incorrect, attr, model, permutation.
        - Print tables.
        - Insight: If incorrect avg conf < correct, model is calibrated.
        """
        if not results['all_comparisons']:
            self.logger.info("No comparisons for confidence analysis.")
            return
        
        df = pd.DataFrame(results['all_comparisons'])
        
        # Overall
        avg_conf = df['conf'].mean()
        med_conf = df['conf'].median()
        print("\nOverall Confidence Stats:")
        print(f"Average: {avg_conf:.2f}, Median: {med_conf:.2f}")
        
        # By correct/incorrect
        conf_by_correct = df.groupby('is_correct')['conf'].agg(['mean', 'median', 'count'])
        print("\nConfidence by Correctness:")
        print(conf_by_correct)
        
        # By attr and correctness
        conf_by_attr_correct = df.groupby(['attr', 'is_correct'])['conf'].agg(['mean', 'median', 'count'])
        print("\nConfidence by Attribute and Correctness:")
        print(conf_by_attr_correct)
        
        # By model and correctness
        conf_by_model_correct = df.groupby(['model', 'is_correct'])['conf'].agg(['mean', 'median', 'count'])
        print("\nConfidence by Model and Correctness:")
        print(conf_by_model_correct)
        
        # By permutation and correctness
        conf_by_perm_correct = df.groupby(['permutation', 'is_correct'])['conf'].agg(['mean', 'median', 'count'])
        print("\nConfidence by Permutation and Correctness:")
        print(conf_by_perm_correct)

    def compute_advanced_metrics(self, results):
        """
        Compute precision, recall, F1 from confusion matrices (macro avg, multi-class).
        - Treat labels as classes (e.g., 'low', 'high', 'na', 'missing').
        - Print per attribute.
        """

        def manual_precision_recall_f1(y_true, y_pred):
            classes = sorted(set(y_true + y_pred))
            per_class_metrics = {}
            precs = []
            recs = []
            f1s = []
            for c in classes:
                tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
                fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
                fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)
                prec = tp / (tp + fp) if tp + fp > 0 else 0
                rec = tp / (tp + fn) if tp + fn > 0 else 0
                f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
                per_class_metrics[c] = {'precision': prec, 'recall': rec, 'f1': f1, 'support': tp + fn}
                precs.append(prec)
                recs.append(rec)
                f1s.append(f1)
            macro_prec = mean(precs) if precs else 0
            macro_rec = mean(recs) if recs else 0
            macro_f1 = mean(f1s) if f1s else 0
            return per_class_metrics, macro_prec, macro_rec, macro_f1
        
        macro_precs = []
        macro_recs = []
        macro_f1s = []

        for attr, conf in results['confusion_per_attr'].items():
            y_true = []
            y_pred = []
            for gt, pred_dict in conf.items():
                for pred, count in pred_dict.items():
                    y_true.extend([gt] * count)
                    y_pred.extend([pred] * count)
            
            if not y_true:
                continue
            
            per_class_metrics, prec, rec, f1 = manual_precision_recall_f1(y_true, y_pred)
            macro_precs.append(prec)
            macro_recs.append(rec)
            macro_f1s.append(f1)

            print(f"\nAdvanced Metrics for {attr}:")
            for cls, metrics in per_class_metrics.items():
                print(f"  Class '{cls}' (support: {metrics['support']}): Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, F1={metrics['f1']:.2f}")
            print(f"Macro Precision: {prec:.2f}")
            print(f"Macro Recall: {rec:.2f}")
            print(f"Macro F1: {f1:.2f}")
            
        if macro_precs:
            overall_prec = sum(macro_precs) / len(macro_precs)
            overall_rec = sum(macro_recs) / len(macro_recs)
            overall_f1 = sum(macro_f1s) / len(macro_f1s)
            print("\nOverall Macro-Averaged Metrics (Across All Attributes):")
            print(f"Precision: {overall_prec:.2f}")
            print(f"Recall: {overall_rec:.2f}")
            print(f"F1-Score: {overall_f1:.2f}")
            

    def run(self) -> dict:
        """
        Run the full calculation.
        Add EDA, charts, mismatch analysis, confidence analysis, advanced metrics.
        """
        self.load_data()
        first_rows = self.preprocess_csv()
        eda_results = self.perform_eda(first_rows)  
        accuracies = self.compute_accuracies(first_rows)
        self.generate_charts(accuracies) 
        self.analyze_mismatches(accuracies)
        self.analyze_confidences(accuracies) 
        self.compute_advanced_metrics(accuracies)  
        return {**accuracies, 'eda': eda_results}  

if __name__ == "__main__":
    
    csv_path = "simulations/final_results/phase_2/profile_rag_ie_kb/autonomous_conversation_history.csv"
    json_path = "data/victim_profile/victim_details.json"  
    
    calculator = ProfileAccuracyCalculator(csv_path, json_path)
    results = calculator.run()
    
    print("Per Conversation Accuracies:")
    for convo_id, info in sorted(results['per_conversation'].items()):
        print(f"Conversation {convo_id} (Profile {info['profile_id']}, Model {info['police_llm_model']}): {info['accuracy']:.2f} ({info['correct']}/{info['total']})")
    
    print(f"\nOverall Accuracy: {results['overall_accuracy']:.2f}")
    
    print("\nAccuracy by Police LLM Model:")
    for model, acc in sorted(results['by_police_model'].items()):
        print(f"{model}: {acc:.2f}")
    
    print("\nAccuracy by Permutation (GT):")
    for seg, acc in sorted(results['by_permutation'].items()):
        print(f"{seg}: {acc:.2f}")
    
    print("\nAccuracy by Police LLM Model and Permutation (GT):")
    for model, seg_accs in sorted(results['by_police_model_and_permutation'].items()):
        print(f"Model: {model}")
        for seg, acc in sorted(seg_accs.items()):
            print(f"  - {seg}: {acc:.2f}")
    
    # Print per-attribute accuracies
    print("\nOverall Per-Attribute Accuracy:")
    for attr, acc in results['overall_per_attr_acc'].items():
        print(f"{attr}: {acc:.2f}")
    
    print("\nPer-Attribute Accuracy by Model:")
    for model, attr_accs in results['by_model_per_attr_acc'].items():
        print(f"Model: {model}")
        for attr, acc in attr_accs.items():
            print(f"  - {attr}: {acc:.2f}")
    
    # Print confusion matrices
    print("\nConfusion Matrices:")
    for attr, conf in results['confusion_per_attr'].items():
        print(f"{attr}:")
        for gt, pred_dict in conf.items():
            print(f"  Actual {gt}: {dict(pred_dict)}")