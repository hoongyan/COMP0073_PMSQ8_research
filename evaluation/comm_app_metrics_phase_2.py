import sys
import os
import pandas as pd
import json
from collections import defaultdict
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.linear_model import LinearRegression

class CommAppScoreCalculator:
    """Class to calculate communication appropriateness scores from CSV and ground truth JSON for phase 2 evaluations."""
    
    def __init__(self, ie_csv_path: str, gt_json_path: str):
        self.ie_csv_path = ie_csv_path
        self.gt_json_path = gt_json_path
        self.ie_df = None
        self.gt_data = None
        self.logger = logging.getLogger("ie_metrics")
        logging.basicConfig(level=logging.INFO)

    def load_data(self):
        """Load CSV and ground truth JSON data."""
        
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
        """Preprocess data by grouping conversations and computing normalized scores."""
        
        # Group by conversation_id
        grouped = self.ie_df.groupby('conversation_id')
        preprocessed = {}
        for convo_id, group in grouped:
            profile_id = int(group['profile_id'].iloc[0])
            
            if profile_id not in self.gt_data:
                self.logger.warning(f"Skipping convo {convo_id}: Profile {profile_id} not in GT.")
                continue
            
            gt = self.gt_data[profile_id]
            gt_user_profile = gt['user_profile']
            scam_type = gt['scam_details']['scam_type']
            
            # Determine segment 
            tech = gt_user_profile.get('tech_literacy', 'unknown').lower()
            lang = gt_user_profile.get('language_proficiency', 'unknown').lower()
            emo = gt_user_profile.get('emotional_state', 'unknown').lower()
            seg = f"{tech}_{lang}_{emo}"
            
            # Compute per-turn overall scores (normalized)
            group['overall'] = (group['tech_literacy'] + group['language_proficiency'] + group['emotional_state']) / 3
            group['overall_norm'] = (group['overall'] - 1) / 4
            
            overall_norms = group['overall_norm'].dropna().tolist() 
            
            if not overall_norms:
                continue  
        
            # Compute raw means
            mean_tech = group['tech_literacy'].mean()
            mean_lang = group['language_proficiency'].mean()
            mean_emo = group['emotional_state'].mean()
            mean_overall = (mean_tech + mean_lang + mean_emo) / 3 if not np.isnan(mean_tech + mean_lang + mean_emo) else np.nan
            
            # Normalize (1-5 scale to 0-1)
            mean_tech_norm = (mean_tech - 1) / 4 if not np.isnan(mean_tech) else np.nan
            mean_lang_norm = (mean_lang - 1) / 4 if not np.isnan(mean_lang) else np.nan
            mean_emo_norm = (mean_emo - 1) / 4 if not np.isnan(mean_emo) else np.nan
            mean_overall_norm = (mean_overall - 1) / 4 if not np.isnan(mean_overall) else np.nan
            min_overall_norm = min(overall_norms) if overall_norms else np.nan
            max_overall_norm = max(overall_norms) if overall_norms else np.nan
            
            # Get LLM model
            llm_model = group['police_llm_model'].iloc[0]
            
            preprocessed[convo_id] = {
                'profile_id': profile_id,
                'gt_user_profile': gt_user_profile,
                'mean_tech_norm': mean_tech_norm,
                'mean_lang_norm': mean_lang_norm,
                'mean_emo_norm': mean_emo_norm,
                'mean_overall_norm': mean_overall_norm,
                'min_overall_norm': min_overall_norm,
                'max_overall_norm': max_overall_norm,
                'length': len(group),
                'overall_norms': overall_norms,
                'segment': seg,
                'scam_type': scam_type,
                'llm_model': llm_model
            }
        self.logger.info(f"Preprocessed {len(preprocessed)} conversations.")
        return preprocessed

    
    def compute_metrics(self, preprocessed):
        """Compute aggregated metrics like means and stds across groupings."""
        
        indicators = ['tech_literacy', 'language_proficiency', 'emotional_state', 'overall']
        ind_to_key = {
            'tech_literacy': 'mean_tech_norm',
            'language_proficiency': 'mean_lang_norm',
            'emotional_state': 'mean_emo_norm',
            'overall': 'mean_overall_norm'
        }
        
        # Collect scores for all groupings
        overall_scores = defaultdict(list)
        by_segment_scores = defaultdict(lambda: defaultdict(list))
        by_llm_scores = defaultdict(lambda: defaultdict(list))
        by_scam_type_scores = defaultdict(lambda: defaultdict(list))
        by_llm_segment_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        by_llm_scam_type_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        by_llm_lengths = defaultdict(list)
        
        for data in preprocessed.values():
          
            by_llm_lengths[data['llm_model']].append(data['length'])
            
            for ind in indicators:
                score = data[ind_to_key[ind]]
                if not np.isnan(score):
                    overall_scores[ind].append(score)
                    by_segment_scores[data['segment']][ind].append(score)
                    by_llm_scores[data['llm_model']][ind].append(score)
                    by_scam_type_scores[data['scam_type']][ind].append(score)
                    by_llm_segment_scores[data['llm_model']][data['segment']][ind].append(score)
                    by_llm_scam_type_scores[data['llm_model']][data['scam_type']][ind].append(score)
        
        # Compute mean/std/count
        def compute_mean_std(scores_dict):
            return {k: {'mean': np.mean(v) if v else 0.0, 'std': np.std(v) if v else 0.0, 'count': len(v)} for k, v in scores_dict.items()}
        
        def compute_nested_mean_std(nested_dict):
            return {outer: compute_mean_std(inner) for outer, inner in nested_dict.items()}
        
        def compute_double_nested_mean_std(double_nested):
            return {outer: compute_nested_mean_std(inner) for outer, inner in double_nested.items()}
        
        # Compute stats for turns per model
        by_llm_turns = compute_mean_std(by_llm_lengths)
        
        return {
            'overall': compute_mean_std(overall_scores),
            'by_segment': compute_nested_mean_std(by_segment_scores),
            'by_llm': compute_nested_mean_std(by_llm_scores),
            'by_scam_type': compute_nested_mean_std(by_scam_type_scores),
            'by_llm_segment': compute_double_nested_mean_std(by_llm_segment_scores),
            'by_llm_scam_type': compute_double_nested_mean_std(by_llm_scam_type_scores),
            'by_llm_turns': by_llm_turns,
          
            'raw_scores': {
                'overall': overall_scores,
                'by_segment': by_segment_scores,
                'by_llm': by_llm_scores,
                'by_scam_type': by_scam_type_scores,
                'by_llm_segment': by_llm_segment_scores
            }
        }

    def generate_appropriateness_charts(self, preprocessed):
        """Generate line charts for appropriateness scores over sorted conversations."""
        
        indicators = ['overall', 'tech_literacy', 'language_proficiency', 'emotional_state']
        ind_to_key = {
            'tech_literacy': 'mean_tech_norm',
            'language_proficiency': 'mean_lang_norm',
            'emotional_state': 'mean_emo_norm',
            'overall': 'mean_overall_norm'
        }
        
        plot_data = pd.DataFrame([
            {'conversation_id': convo_id, **{ind_to_key[ind]: data[ind_to_key[ind]] for ind in indicators}, 'llm_model': data['llm_model']}
            for convo_id, data in preprocessed.items()
        ])
        plot_data = plot_data.dropna(subset=[ind_to_key['overall']])  # Drop NaN
        
        base_dir = Path('evaluation/results/communication_appropriateness/phase_2/charts')
        base_dir.mkdir(parents=True, exist_ok=True)
        
        models = sorted(plot_data['llm_model'].unique())  
        self.logger.info(f"Unique models: {models}")  
        import matplotlib.colors as mcolors
        color_list = list(mcolors.TABLEAU_COLORS.values())
      
        colors = {model: color_list[i % len(color_list)] for i, model in enumerate(models)}
        
        # Custom colors for specified models
        for model in models:
            model_lower = model.lower()
            if 'gpt-4o-mini' in model_lower:
                colors[model] = '#2CA02C'  # Green for ChatGPT      
            elif 'mistral' in model_lower:
                colors[model] = '#D62728'  # Red for Mistral     
            elif model_lower == 'qwen2.5:7b':
                colors[model] = '#FF7F0E'  # Orange for Qwen
            elif model_lower == 'granite3.2:8b':
                colors[model] = '#1F77B4'  # Blue for Granite
               
        for ind in indicators:
            fig, ax = plt.subplots(figsize=(14, 7))
            key = ind_to_key[ind]
            
            for model in models:
                model_data = plot_data[plot_data['llm_model'] == model].sort_values('conversation_id')
                if len(model_data) < 2:
                    self.logger.warning(f"Skipping model {model}: insufficient data points ({len(model_data)})")
                    continue
                
                x = list(range(1, len(model_data) + 1))
                y = model_data[key].values
                color = colors[model]
                
                ax.plot(x, y, label=model, color=color, alpha=0.3, linewidth=1)
                X = np.array(x).reshape(-1, 1)
                reg = LinearRegression().fit(X, y)
                trend = reg.predict(X)
                ax.plot(x, trend, linestyle='--', color=color, label=f'{model} Trend', linewidth=2)
            
            ax.set_title(f'Appropriateness Score Over Sorted Conversations ({ind.capitalize()})', fontsize=20)  
            ax.set_xlabel('Conversation Index (Sorted by ID)', fontsize=20)  
            ax.set_ylabel('Normalized Score', fontsize=20)  
            ax.set_ylim(0.2, 1)  
            ax.tick_params(axis='both', labelsize=18)  
            ax.legend(fontsize=14, ncol=2)
            ax.grid(True)
            self.save_figure(fig, str(base_dir / f'appropriateness_over_convos_combined_{ind}.png'))
            
    def generate_charts(self, metrics, preprocessed):
        """Generate bar, boxplot, and heatmap charts from metrics and preprocessed data."""
        
        base_dir = Path('evaluation/results/communication_appropriateness/phase_2/charts')
        base_dir.mkdir(parents=True, exist_ok=True)
        
        indicators = ['tech_literacy', 'language_proficiency', 'emotional_state']
        all_indicators = ['overall'] + indicators
        ind_to_key = {
            'tech_literacy': 'mean_tech_norm',
            'language_proficiency': 'mean_lang_norm',
            'emotional_state': 'mean_emo_norm',
            'overall': 'mean_overall_norm'
        }
        
        overall_scores = defaultdict(list)
        for data in preprocessed.values():
            for ind in all_indicators:
                score = data.get(ind_to_key[ind])
                if score is not None and not np.isnan(score):  
                    overall_scores[ind].append(score)
        
        # Define segment order and maps 
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
        
        # Bar: Overall and per-indicator means
        means = [metrics['overall'][ind]['mean'] for ind in all_indicators]
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(all_indicators, means, color='skyblue', width=0.4)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=16)
        ax.set_title('Mean Appropriateness Score: Overall and Per-Attribute', fontsize=20)
        ax.set_ylabel('Normalized Score', fontsize=16)
        ax.set_ylim(0, 1.1)
        ax.tick_params(axis='x', labelsize=16, rotation=0)
        ax.tick_params(axis='y', labelsize=16)
        self.save_figure(fig, str(base_dir / 'mean_bar_overall_per_indicator.png'))
        
        # Boxplot: Distribution by indicator
        data = [overall_scores[ind] for ind in indicators]  # Exclude overall for box
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot(data, patch_artist=True, boxprops=dict(facecolor='skyblue'))
        ax.set_xticklabels(indicators, fontsize=16)
        ax.set_title('Distribution of Appropriateness Scores by Profile Attributes', fontsize=20)
        ax.set_ylabel('Normalized Score', fontsize=16)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='y', labelsize=16)
        self.save_figure(fig, str(base_dir / 'boxplot_by_indicator.png'))
        
        # Bar: By segment (overall)
        all_segs = set(metrics['by_segment'].keys())
        segs = [s for s in desired_order if s in all_segs]
        short_segs = [segment_map[s] for s in segs]
        means = [metrics['by_segment'][s]['overall']['mean'] for s in segs]
        fig, ax = plt.subplots(figsize=(12, 10))
        bars = ax.bar(short_segs, means, color='skyblue', width=0.4)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=18)
        ax.set_title('Mean Appropriateness Score by Victim Segment', fontsize=20)
        ax.set_ylabel('Normalized Score', fontsize=18)
        ax.set_ylim(0, 1.1)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        self.save_figure(fig, str(base_dir / 'mean_bar_by_segment.png'))
        
        # Bar: By scam_type (overall)
        scam_types = sorted(metrics['by_scam_type'].keys())
        means = [metrics['by_scam_type'][st]['overall']['mean'] for st in scam_types]
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.bar(scam_types, means, color='skyblue', width=0.4)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=18)
        ax.set_title('Mean Appropriateness Score by Scam Type', fontsize=20)
        ax.set_ylabel('Normalized Score', fontsize=18)
        ax.set_ylim(0, 1.1)
        ax.tick_params(axis='x', labelsize=18, rotation=0)
        ax.tick_params(axis='y', labelsize=18)
        self.save_figure(fig, str(base_dir / 'mean_bar_by_scam_type.png'))
        
        # Heatmap: Model vs Segment (overall score)
        models = sorted(metrics['by_llm_segment'].keys())
        segs = [s for s in desired_order if s in set(s for m in models for s in metrics['by_llm_segment'][m])]
        short_segs = [segment_map[s] for s in segs]
        data = np.array([[metrics['by_llm_segment'][m][s]['overall']['mean'] if s in metrics['by_llm_segment'][m] else np.nan for s in segs] for m in models])
        std_data = np.array([[metrics['by_llm_segment'][m][s]['overall']['std'] if s in metrics['by_llm_segment'][m] else np.nan for s in segs] for m in models])
        fig, ax = plt.subplots(figsize=(15, 6))
        im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=1)
        cbar = plt.colorbar(im)
        cbar.ax.tick_params(labelsize=18)
        ax.set_xticks(range(len(short_segs)))
        ax.set_xticklabels(short_segs, rotation=0, fontsize=18)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=18)
        ax.set_title('Mean Appropriateness Score: Model vs Victim Segment', fontsize=20)
        for i in range(len(models)):
            for j in range(len(segs)):
                if not np.isnan(data[i, j]):
                    text = f'{data[i, j]:.2f}\n±{std_data[i, j]:.2f}'
                else:
                    text = '-'
                ax.text(j, i, text, ha='center', va='center', color='black', fontsize=18)
        self.save_figure(fig, str(base_dir / 'heatmap_model_vs_segment.png'))
        
        # Heatmap: Model vs Indicator
        data = np.array([[metrics['by_llm'][m][ind]['mean'] for ind in indicators] for m in models]).T
        fig, ax = plt.subplots(figsize=(15, 6))
        im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        cbar = plt.colorbar(im)
        cbar.ax.tick_params(labelsize=18)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=0, fontsize=18)
        ax.set_yticks(range(len(indicators)))
        ax.set_yticklabels(indicators, fontsize=18)
        ax.set_title('Mean Appropriateness Score: Profile Attribute vs Model', fontsize=20)
        for i in range(len(indicators)):
            for j in range(len(models)):
                ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', color='black', fontsize=18)
        plt.subplots_adjust(left=0.25)
        self.save_figure(fig, str(base_dir / 'heatmap_indicator_vs_model.png'))
        
        # Heatmap: Indicator vs Segment
        data = np.array([[metrics['by_segment'][s][ind]['mean'] for ind in indicators] for s in segs]).T
        fig, ax = plt.subplots(figsize=(15, 6))
        im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        cbar = plt.colorbar(im,pad=0.10)
        cbar.ax.tick_params(labelsize=18)
        ax.set_xticks(range(len(short_segs)))
        ax.set_xticklabels(short_segs, rotation=0, fontsize=18)
        ax.set_yticks(range(len(indicators)))
        ax.set_yticklabels(indicators, fontsize=18)
        ax.set_title('Mean Appropriateness Score: Profile Attribute vs Victim Segment', fontsize=20)
        for i in range(len(indicators)):
            for j in range(len(segs)):
                ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', color='black', fontsize=18)
        plt.subplots_adjust(left=0.25)
        self.save_figure(fig, str(base_dir / 'heatmap_indicator_vs_segment.png'))
        
        # Heatmap: Model vs Scam Type (overall)
        scam_types = sorted(metrics['by_scam_type'].keys())
        data = np.array([[metrics['by_llm_scam_type'][m].get(st, {}).get('overall', {}).get('mean', np.nan) 
                  for st in scam_types] for m in models])
        fig, ax = plt.subplots(figsize=(15, 6))
        im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        cbar = plt.colorbar(im)
        cbar.ax.tick_params(labelsize=18)
        ax.set_xticks(range(len(scam_types)))
        ax.set_xticklabels(scam_types, rotation=0, fontsize=18)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=18)
        ax.set_title('Mean Appropriateness Score: Model vs Scam Type', fontsize=20)
        for i in range(len(models)):
            for j in range(len(scam_types)):
                ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', color='black', fontsize=18)
        self.save_figure(fig, str(base_dir / 'heatmap_model_vs_scam_type.png'))

    def generate_length_impact_charts(self, preprocessed):
        """Generate scatter and line charts for impact of conversation length on scores."""
        
        base_dir = Path('evaluation/results/communication_appropriateness/phase_2/charts')
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect data for scatters
        lengths = []
        means = []
        mins = []
        maxs = []
        for data in preprocessed.values():
            if not np.isnan(data['mean_overall_norm']):
                lengths.append(data['length'])
                means.append(data['mean_overall_norm'])
                mins.append(data['min_overall_norm'])
                maxs.append(data['max_overall_norm'])
        
        if not lengths:
            self.logger.warning("No data for length impact charts.")
            return
        
        # Scatter: Length vs Mean
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(lengths, means, color='blue', label='Mean Norm')
        if len(lengths) > 1:
            X = np.array(lengths).reshape(-1, 1)
            y = np.array(means)
            reg = LinearRegression().fit(X, y)
            trend = reg.predict(X)
            ax.plot(lengths, trend, color='red', label='Trend')
        ax.set_title('Conversation Length vs Mean Appropriateness Score')
        ax.set_xlabel('Length (Turns)')
        ax.set_ylabel('Normalized Mean Score')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True)
        self.save_figure(fig, str(base_dir / 'scatter_length_vs_mean.png'))
        
        # Scatter: Length vs Min
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(lengths, mins, color='green', label='Min Norm')
        if len(lengths) > 1:
            X = np.array(lengths).reshape(-1, 1)
            y = np.array(mins)
            reg = LinearRegression().fit(X, y)
            trend = reg.predict(X)
            ax.plot(lengths, trend, color='red', label='Trend')
        ax.set_title('Conversation Length vs Min Appropriateness Score')
        ax.set_xlabel('Length (Turns)')
        ax.set_ylabel('Normalized Min Score')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True)
        self.save_figure(fig, str(base_dir / 'scatter_length_vs_min.png'))
        
        # Scatter: Length vs Max
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(lengths, maxs, color='orange', label='Max Norm')
        if len(lengths) > 1:
            X = np.array(lengths).reshape(-1, 1)
            y = np.array(maxs)
            reg = LinearRegression().fit(X, y)
            trend = reg.predict(X)
            ax.plot(lengths, trend, color='red', label='Trend')
        ax.set_title('Conversation Length vs Max Appropriateness Score')
        ax.set_xlabel('Length (Turns)')
        ax.set_ylabel('Normalized Max Score')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True)
        self.save_figure(fig, str(base_dir / 'scatter_length_vs_max.png'))

        models = sorted(set(data['llm_model'] for data in preprocessed.values()))  # Sort for consistent order
        preprocessed_by_model = {model: [data for data in preprocessed.values() if data['llm_model'] == model] for model in models}


        color_list = list(mcolors.TABLEAU_COLORS.values())
        colors = {model: color_list[i % len(color_list)] for i, model in enumerate(models)}
        # Custom colors for specified models
        for model in models:
            model_lower = model.lower()
            if 'gpt-4o-mini' in model_lower:
                colors[model] = '#2CA02C'  # Green for ChatGPT      
            elif 'mistral' in model_lower:
                colors[model] = '#D62728'  # Red for Mistral     
            elif model_lower == 'qwen2.5:7b':
                colors[model] = '#FF7F0E'  # Orange for Qwen
            elif model_lower == 'granite3.2:8b':
                colors[model] = '#1F77B4'  # Blue for Granite

        max_length = max([data['length'] for data in preprocessed.values()]) if preprocessed else 0

        # Compute avg and count per turn per model
        avg_per_turn_by_model = {model: [] for model in models}
        count_per_turn_by_model = {model: [] for model in models}

        for model in models:
            model_data = preprocessed_by_model[model]
            for turn in range(1, max_length + 1):
                scores = [data['overall_norms'][turn-1] for data in model_data if len(data['overall_norms']) >= turn]
                if scores:
                    avg_per_turn_by_model[model].append(np.mean(scores))
                    count_per_turn_by_model[model].append(len(scores))
                else:
                    avg_per_turn_by_model[model].append(np.nan)
                    count_per_turn_by_model[model].append(0)
                    
        fig, ax = plt.subplots(figsize=(12, 6))

        for model in models:
            x = range(1, max_length + 1)
            y = avg_per_turn_by_model[model]
            color = colors[model]
            
            # Plot score line
            ax.plot(x, y, label=model, color=color, alpha=0.7, linewidth=1)
            
            if len(y) > 1 and not all(np.isnan(val) for val in y):
                valid_x = [pos for pos, val in zip(x, y) if not np.isnan(val)]
                valid_y = [val for val in y if not np.isnan(val)]
                if len(valid_x) > 1:
                    X = np.array(valid_x).reshape(-1, 1)
                    reg = LinearRegression().fit(X, valid_y)
                    trend = reg.predict(X)
                    ax.plot(valid_x, trend, linestyle='--', color=color, label=f'{model} Trend', linewidth=2)

        ax.set_title('Average Appropriateness Score by Turn Position (Per Model)', fontsize=20)
        ax.set_xlabel('Turn Position', fontsize=20)
        ax.set_ylabel('Normalized Avg Score', fontsize=20)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='both', labelsize=20)
        ax.legend(fontsize=16, ncol=2)  
        ax.grid(True)

        overall_count_per_turn = [sum(count_per_turn_by_model[model][turn-1] for model in models) for turn in range(1, max_length + 1)]
        ax2 = ax.twinx()
        ax2.plot(range(1, max_length + 1), overall_count_per_turn, color='gray', linestyle='--', label='Total Convo Count')
        ax2.set_ylabel('Number of Conversations', fontsize=20)
        ax2.tick_params(axis='y', labelsize=20)
        ax2.legend(loc='lower right', fontsize=16)

        self.save_figure(fig, str(base_dir / 'line_avg_by_turn_position_per_model.png'))
        
    
    def save_figure(self, fig, filepath: str) -> None:
        """Save matplotlib figure to the specified filepath."""
        try:
            fig.savefig(filepath)
            self.logger.info(f"Saved figure to '{filepath}'")
        except Exception as e:
            self.logger.error(f"Failed to save figure to '{filepath}': {e}")
        finally:
            plt.close(fig)

    def run(self):
        """Run the full pipeline: load data, preprocess, compute metrics, and generate charts."""
        
        self.load_data()
        preprocessed = self.preprocess()
        metrics = self.compute_metrics(preprocessed)
        self.generate_appropriateness_charts(preprocessed)
        self.generate_charts(metrics, preprocessed)
        self.generate_length_impact_charts(preprocessed)
        return metrics

if __name__ == "__main__":
    ie_csv = "evaluation/results/communication_appropriateness/phase_2/evaluated_autonomous_conversations_ie.csv"
    victim_detail_json = "data/victim_profile/victim_details.json"
    
    calculator = CommAppScoreCalculator(ie_csv, victim_detail_json)
    results = calculator.run()
    print(results)

    print("\nAverage number of turns per model:")
    if 'by_llm_turns' in results:
        for model, stats in sorted(results['by_llm_turns'].items()):
            print(f"{model}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, count={stats['count']}")
    else:
        print("No 'by_llm_turns' data available.")