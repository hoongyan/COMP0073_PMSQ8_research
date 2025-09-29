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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer

# Import settings and logger setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.settings import get_settings
from config.logging_config import setup_logger

class UpsertStrategyCalculator:
    def __init__(self, history_csv_path: str, strategies_csv_path: str):
        self.history_csv_path = history_csv_path
        self.strategies_csv_path = strategies_csv_path
        self.history_df = None
        self.strategies_df = None
        self.convo_to_model = {}
        self.upserts = []
        self.settings = get_settings()
        self.logger = setup_logger("upsert_metrics", self.settings.log.subdirectories["evaluation"])

    def load_data(self):
        # Load csv
        self.history_df = pd.read_csv(self.history_csv_path, index_col=False, engine='python', on_bad_lines='warn')
      
        self.history_df['conversation_id'] = pd.to_numeric(self.history_df['conversation_id'], errors='coerce')
        self.history_df = self.history_df.dropna(subset=['conversation_id'])
        self.history_df['conversation_id'] = self.history_df['conversation_id'].astype('int')
        self.logger.info(f"Loaded History CSV with {len(self.history_df)} rows.")
        self.logger.info(f"History columns: {self.history_df.columns.tolist()}")
        self.logger.info(f"Sample rag_upsert (row 0): {self.history_df['rag_upsert'].iloc[0] if not self.history_df.empty else 'Empty'}")
        self.logger.info(f"Sample upserted_strategy (row 0): {self.history_df['upserted_strategy'].iloc[0] if not self.history_df.empty else 'Empty'}")

        # Collect upserted strategies
        self.history_df['upserted_strategy'] = self.history_df['upserted_strategy'].fillna('{}')
        upserts_rows = self.history_df[(self.history_df['rag_upsert'] == True) & (self.history_df['upserted_strategy'] != '{}')]
        for _, row in upserts_rows.iterrows():
            try:
                strat = json.loads(row['upserted_strategy'])
                strat['model'] = row['police_llm_model']
                self.upserts.append(strat)
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse upserted_strategy: {row['upserted_strategy']}")

        self.logger.info(f"Collected {len(self.upserts)} upserted strategies.")

        grouped_history = self.history_df.groupby('conversation_id')
        for convo_id, group in grouped_history:
            models = group['police_llm_model'].unique()
            if len(models) == 1:
                self.convo_to_model[convo_id] = models[0]
            else:
                self.logger.warning(f"Multiple models for convo {convo_id}: {models}. Using the first one.")
                self.convo_to_model[convo_id] = models[0]
        self.logger.info(f"Built model mapping for {len(self.convo_to_model)} conversations.")

        # Load Strategies CSV
        self.strategies_df = pd.read_csv(self.strategies_csv_path, index_col=False, engine='python', on_bad_lines='warn')
        self.strategies_df['timestamp'] = pd.to_datetime(self.strategies_df['timestamp'], utc=True, errors='coerce')
        self.strategies_df = self.strategies_df.sort_values('timestamp')
        self.logger.info(f"Loaded Strategies CSV with {len(self.strategies_df)} rows.")

    def preprocess(self):
        # Group by conversation_id
        grouped = self.history_df.groupby('conversation_id')
        preprocessed = {}
        for convo_id, group in grouped:
            if convo_id not in self.convo_to_model:
                self.logger.warning(f"No model mapping for convo {convo_id}. Skipping.")
                continue
            model = self.convo_to_model[convo_id]

            # Count the number of upserts: where rag_upsert is True and upserted_strategy is not empty
            group['upserted_strategy'] = group['upserted_strategy'].fillna('{}')
            upserts = group[(group['rag_upsert'] == True) & (group['upserted_strategy'] != '{}')]
            num_upserts = len(upserts)

            preprocessed[convo_id] = {
                'model': model,
                'num_upserts': num_upserts
            }
        self.logger.info(f"Preprocessed {len(preprocessed)} conversations.")
        return preprocessed

    def compute_metrics(self, preprocessed):
        by_model_counts = defaultdict(list)
        overall_upserts = []
        per_convo_upserts = {}
        models_per_convo = {}

        for convo_id, data in preprocessed.items():
            num = data['num_upserts']
            model = data['model']
            by_model_counts[model].append(num)
            overall_upserts.append(num)
            per_convo_upserts[convo_id] = num
            models_per_convo[convo_id] = model

        def safe_mean(lst):
            return np.mean(lst) if lst else 0.0

        def safe_sum(lst):
            return sum(lst) if lst else 0

        metrics = {
            'overall_average_upserts': safe_mean(overall_upserts),
            'overall_total_upserts': safe_sum(overall_upserts),
            'by_model_average': {m: safe_mean(counts) for m, counts in by_model_counts.items()},
            'by_model_total': {m: safe_sum(counts) for m, counts in by_model_counts.items()},
            'by_model_num_convos': {m: len(counts) for m, counts in by_model_counts.items()},
            'per_convo_upserts': per_convo_upserts,
            'models_per_convo': models_per_convo,
            'by_model_counts': dict(by_model_counts)
        }
        return metrics

    def compute_kb_dynamics(self, metrics):
        # Total KB size: unique IDs
        total_kb = len(self.strategies_df['id'].unique())

        # Addition rate: (num upserts / total KB) * 100 (focus on this as growth proxy)
        num_upserts = len(self.upserts)
        addition_rate_pct = (num_upserts / total_kb) * 100 if total_kb > 0 else 0.0

        # Pruning stats
        min_id = self.strategies_df['id'].min()
        max_id = self.strategies_df['id'].max()
        expected_count = max_id - min_id + 1 if pd.notna(min_id) and pd.notna(max_id) else 0
        current_count = total_kb
        pruned_count = expected_count - current_count if expected_count > 0 else 0
        pruning_rate_pct = (pruned_count / expected_count) * 100 if expected_count > 0 else 0.0

        # Retrieval stats for remaining strategies
        retrieval_min = self.strategies_df['retrieval_count'].min()
        retrieval_max = self.strategies_df['retrieval_count'].max()
        retrieval_mean = self.strategies_df['retrieval_count'].mean()

        # Diversity: Unique strategy types
        unique_strategy_types = len(self.strategies_df['strategy_type'].unique())

        metrics['total_kb_size'] = total_kb
        metrics['addition_rate_pct'] = addition_rate_pct
        metrics['pruned_count'] = pruned_count
        metrics['pruning_rate_pct'] = pruning_rate_pct
        metrics['retrieval_min'] = retrieval_min
        metrics['retrieval_max'] = retrieval_max
        metrics['retrieval_mean'] = retrieval_mean
        metrics['unique_strategy_types'] = unique_strategy_types

        self.logger.info(f"KB Dynamics: Addition rate {addition_rate_pct:.2f}%, Pruning rate {pruning_rate_pct:.2f}%, Unique types {unique_strategy_types}, Retrieval min/max/mean {retrieval_min}/{retrieval_max}/{retrieval_mean:.2f}")

        return metrics

    def cluster_strategies(self, metrics):
        embedder = SentenceTransformer('all-MiniLM-L6-v2')

        # Helper function to perform clustering
        def perform_clustering(features, prefix, max_k=10):
            if len(features) < 2:
                metrics[f'{prefix}_optimal_k'] = 0
                metrics[f'{prefix}_clustering_health'] = "N/A - Insufficient data"
                return

            # Elbow method
            inertias = []
            for k in range(1, min(max_k, len(features)) + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(features)
                inertias.append(kmeans.inertia_)

            # Plot elbow
            upsert_dir = Path(self.settings.data.evaluation_results_dir) / 'kb'
            fig, ax = plt.subplots()
            ax.plot(range(1, len(inertias) + 1), inertias, marker='o')
            ax.set_title(f'Elbow Method for {prefix.capitalize()} Clusters')
            ax.set_xlabel('Number of Clusters')
            ax.set_ylabel('Inertia')
            fig.savefig(upsert_dir / f'{prefix}_clusters_elbow.png')
            plt.close(fig)

            # Silhouette scores
            sil_scores = []
            for k in range(2, min(max_k, len(features)) + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(features)
                sil = silhouette_score(features, labels)
                sil_scores.append(sil)

            if sil_scores:
                optimal_k = range(2, len(sil_scores) + 2)[np.argmax(sil_scores)]
            else:
                optimal_k = 1

            # Plot silhouette
            if sil_scores:
                fig, ax = plt.subplots()
                ax.plot(range(2, len(sil_scores) + 2), sil_scores, marker='o')
                ax.set_title(f'Silhouette Scores for {prefix.capitalize()} Clusters')
                ax.set_xlabel('Number of Clusters')
                ax.set_ylabel('Silhouette Score')
                fig.savefig(upsert_dir / f'{prefix}_silhouette.png')
                plt.close(fig)

            clustering_health = "Low - potentially stuck in few similar types" if optimal_k < 3 else "Good - varied clusters indicating diversity"

            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)
            metrics[f'{prefix}_cluster_labels'] = labels.tolist()

            # PCA visualization
            if features.shape[1] > 1:
                pca = PCA(n_components=2)
                reduced = pca.fit_transform(features)
                fig, ax = plt.subplots()
                ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='viridis')
                ax.set_title(f'PCA of {prefix.capitalize()} Clusters (k={optimal_k})')
                ax.set_xlabel('PCA Component 1')
                ax.set_ylabel('PCA Component 2')
                fig.savefig(upsert_dir / f'{prefix}_clusters_pca.png')
                plt.close(fig)

            metrics[f'{prefix}_optimal_k'] = optimal_k
            metrics[f'{prefix}_clustering_health'] = clustering_health
            self.logger.info(f"Clustered {prefix} into {optimal_k} clusters. Health: {clustering_health}")

        kb_strategy_types = self.strategies_df['strategy_type'].dropna().unique().tolist()
        if len(kb_strategy_types) >= 2:
            kb_features = embedder.encode(kb_strategy_types)
            perform_clustering(kb_features, 'kb')
        else:
            metrics['kb_optimal_k'] = 0
            metrics['kb_clustering_health'] = "N/A - Insufficient data for KB"

        if len(self.upserts) < 2:
            metrics['upsert_optimal_k'] = 0
            metrics['upsert_clustering_health'] = "N/A - Insufficient data for upserts"
        else:
            upsert_strategy_types = [s['strategy_type'] for s in self.upserts if 'strategy_type' in s]
            upsert_unique_types = list(set(upsert_strategy_types))
            if len(upsert_unique_types) >= 2:
                upsert_features = embedder.encode(upsert_unique_types)
                perform_clustering(upsert_features, 'upsert')
            else:
                metrics['upsert_optimal_k'] = 0
                metrics['upsert_clustering_health'] = "N/A - Insufficient data for upserts"

        return metrics

    def generate_charts(self, metrics):
        if not metrics['by_model_average']:
            self.logger.warning("No data for charts.")
            return

        base_eval_dir = Path(self.settings.data.evaluation_results_dir)
        upsert_dir = base_eval_dir / 'kb'
        upsert_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Ensured output directory exists: {upsert_dir}")

        cleaned_models = [m.strip().strip("'\"") for m in metrics['by_model_average'].keys()]

        # Bar chart: Average upserts per conversation by model
        models = cleaned_models
        avgs = list(metrics['by_model_average'].values())
        fig, ax = plt.subplots()
        ax.bar(models, avgs, label='Average Upserts')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_title('Average Number of Upserted Strategies per Conversation by Model')
        ax.set_ylabel('Average Upserts')
        for i, v in enumerate(avgs):
            ax.text(i, v, f"{v:.2f}", ha='center', va='bottom')
        fig.savefig(upsert_dir / 'average_upserts_by_model.png')
        plt.close(fig)

        # Bar chart: Total upserts by model
        totals = list(metrics['by_model_total'].values())
        fig, ax = plt.subplots()
        ax.bar(models, totals, label='Total Upserts')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_title('Total Number of Upserted Strategies by Model')
        ax.set_ylabel('Total Upserts')
        for i, v in enumerate(totals):
            ax.text(i, v, f"{v}", ha='center', va='bottom')
        fig.savefig(upsert_dir / 'total_upserts_by_model.png')
        plt.close(fig)

        # Line graph: Upserts per conversation over conversation IDs
        convo_ids = sorted(metrics['per_convo_upserts'].keys())
        upserts = [metrics['per_convo_upserts'][cid] for cid in convo_ids]
        models_seq = [metrics['models_per_convo'][cid].strip().strip("'\"") for cid in convo_ids]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(convo_ids, upserts, marker='o', linestyle='-', color='b', label='Upserts per Conversation')
        ax.set_title('Number of Upserted Strategies per Conversation (Sequential by ID)')
        ax.set_xlabel('Conversation ID')
        ax.set_ylabel('Number of Upserts')
        ax.legend()

        # Indicate model changes with vertical lines and labels
        prev_model = None
        for i, (cid, model) in enumerate(zip(convo_ids, models_seq)):
            if model != prev_model:
                ax.axvline(x=cid, color='r', linestyle='--', alpha=0.5)
                ax.text(cid, max(upserts) * 0.95, model, rotation=90, ha='right', va='top', fontsize=10)
            prev_model = model

        fig.savefig(upsert_dir / 'upserts_per_convo_line.png')
        plt.close(fig)

        # Boxplot: Distribution of upserts per conversation by model
        data = []
        for m_raw, counts in metrics['by_model_counts'].items():
            m = m_raw.strip().strip("'\"")
            for c in counts:
                data.append({'model': m, 'upserts': c})
        df = pd.DataFrame(data)
        if not df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='model', y='upserts', data=df, ax=ax)
            ax.set_title('Distribution of Upserted Strategies per Conversation by Model')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            fig.savefig(upsert_dir / 'boxplot_upserts_by_model.png')
            plt.close(fig)
            
        # Stacked area chart for KB growth over conversations
        convo_ids = sorted(metrics['per_convo_upserts'].keys())
        upserts = [metrics['per_convo_upserts'][cid] for cid in convo_ids]
        models_seq = [metrics['models_per_convo'][cid].strip().strip("'\"") for cid in convo_ids]

        unique_models = []
        seen = set()
        for m in models_seq:
            if m not in seen:
                unique_models.append(m)
                seen.add(m)

        colors = sns.color_palette("husl", len(unique_models))
        model_to_color = dict(zip(unique_models, colors))

        # Compute cumulative contributions per model
        n = len(convo_ids)
        model_cumuls = {m: np.zeros(n) for m in unique_models}
        current = {m: 0 for m in unique_models}
        for i in range(n):
            m = models_seq[i]
            current[m] += upserts[i]
            for mm in unique_models:
                model_cumuls[mm][i] = current[mm]


        x = np.arange(n + 1)
        model_cumuls_ext = {m: np.insert(model_cumuls[m], 0, 0) for m in unique_models}


        fig, ax = plt.subplots(figsize=(12, 6))
        bottom = np.zeros(len(x)) + 18  # Seed as base
        ax.fill_between(x, 0, 18, color='gray', alpha=0.3, label='Seed Strategies')
        for m in unique_models:
            top = bottom + model_cumuls_ext[m]
            ax.fill_between(x, bottom, top, color=model_to_color[m], label=m)
            bottom = top
        ax.plot(x, bottom, 'k-', linewidth=1, label='Total Strategies')
        ax.set_xlabel('Conversations Completed')
        ax.set_ylabel('Total Strategies in KB')
        ax.set_title('Growth of Knowledgebase Over Simulations')
        ax.set_xlim(0, n)
        ax.legend(loc='upper left')
        fig.savefig(upsert_dir / 'kb_growth_stacked.png')
        plt.close(fig)

        cumulative_kb = np.cumsum([1 for _ in self.strategies_df.iterrows()]) 
        unique_types_cum = []
        seen_types = set()
        for _, row in self.strategies_df.iterrows():
            seen_types.add(row['strategy_type'])
            unique_types_cum.append(len(seen_types))

        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.plot(x, bottom, color='b', label='Cumulative KB Size')
        ax1.set_xlabel('Conversations Completed')
        ax1.set_ylabel('KB Size', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_title('KB Growth vs. Strategy Diversity Over Conversations')

        ax2 = ax1.twinx()

        diversity_ext = [0] + unique_types_cum  

        if len(diversity_ext) != len(x):
            from scipy.interpolate import interp1d
            interp_func = interp1d(np.linspace(0, n, len(diversity_ext)), diversity_ext, kind='linear')
            interpolated_diversity = interp_func(np.arange(n+1))
        else:
            interpolated_diversity = diversity_ext
        ax2.plot(x, interpolated_diversity, color='g', label='Cumulative Unique Strategy Types')
        ax2.set_ylabel('Unique Strategy Types', color='g')
        ax2.tick_params(axis='y', labelcolor='g')

        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
        fig.tight_layout()
        fig.savefig(upsert_dir / 'kb_growth_vs_diversity.png')
        plt.close(fig)

        # Bar for addition and pruning rates
        fig, ax = plt.subplots()
        ax.bar(['Addition Rate (%)', 'Pruning Rate (%)'], [metrics['addition_rate_pct'], metrics['pruning_rate_pct']])
        ax.set_title('KB Addition and Pruning Rates')
        ax.set_ylabel('Percentage')
        for i, v in enumerate([metrics['addition_rate_pct'], metrics['pruning_rate_pct']]):
            ax.text(i, v, f"{v:.2f}%", ha='center', va='bottom')
        fig.savefig(upsert_dir / 'kb_rates_bar.png')
        plt.close(fig)

        # Histogram for retrieval counts
        fig, ax = plt.subplots()
        self.strategies_df['retrieval_count'].hist(ax=ax, bins=20)
        ax.set_title('Distribution of Retrieval Counts')
        ax.set_xlabel('Retrieval Count')
        ax.set_ylabel('Frequency')
        fig.savefig(upsert_dir / 'retrieval_counts_hist.png')
        plt.close(fig)

        self.logger.info("Generated charts.")

    def run(self):
        self.load_data()
        preprocessed = self.preprocess()
        metrics = self.compute_metrics(preprocessed)
        metrics = self.compute_kb_dynamics(metrics)  
        metrics = self.cluster_strategies(metrics)
        self.generate_charts(metrics)
        return metrics


if __name__ == "__main__":
    history_csv = "simulations/final_results/phase_2/profile_rag_ie_kb/autonomous_conversation_history.csv"
    strategies_csv = "simulations/finall_results/phase_2/profile_rag_ie_kb/strategies.csv"  

    calculator = UpsertStrategyCalculator(history_csv, strategies_csv)
    results = calculator.run()
    print(results)