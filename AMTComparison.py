#!/usr/bin/env python
# coding: utf-8
"""
AMTComparison: Advanced Multi-Neuron Analysis Comparison Tool

This module provides comprehensive comparison functionality for analyzing and comparing
multiple neurons' analysis outputs from the AMT (Advanced Morphology Tool) pipeline.

Features:
  - Multi-neuron comparison (2 or more neurons)
  - Statistical testing (Mann-Whitney U, Kruskal-Wallis, Cohen's d)
  - Multiple comparison correction (Bonferroni)
  - Temporal dynamics, spatial statistics, and network topology comparisons
  - Colorblind-safe visualizations with significance annotations
  - Comprehensive comparison reports (CSV and multi-panel figures)

Usage:
  Command-line:
    python AMTComparison.py --dirs dir1 dir2 dir3 --labels "WT" "KO" "Rescue" --output results/

  Python API:
    from AMTComparison import ComparativeAnalysis
    ca = ComparativeAnalysis(dirs=['dir1', 'dir2'], labels=['WT', 'KO'])
    ca.run_full_analysis(output_dir='results/')

Author: AMT Development Team
Version: 2.0.0
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal
import matplotlib.patches as mpatches

# ============================================================
# Global Configuration
# ============================================================

# Colorblind-safe palette
CB_COLORS = ["#0173B2", "#DE8F05", "#CC79A7", "#009E73", "#D55E00", "#56B4E9"]

# Visualization settings
sns.set_style("whitegrid")
sns.set_context("talk")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# ============================================================
# Utility Functions
# ============================================================

def load_csv_safe(folder: Union[str, Path], relative_path: str) -> Optional[pd.DataFrame]:
    """
    Safely load a CSV file if it exists.

    Parameters
    ----------
    folder : str or Path
        Base directory path
    relative_path : str
        Relative path from folder to CSV file

    Returns
    -------
    pd.DataFrame or None
        Loaded dataframe or None if file not found or error occurred
    """
    path = Path(folder) / relative_path
    if not path.exists():
        print(f"⚠️  Missing file: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"❌ Error reading {path.name}: {e}")
        return None


def cohen_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size between two groups.

    Parameters
    ----------
    group1 : np.ndarray
        First group values
    group2 : np.ndarray
        Second group values

    Returns
    -------
    float
        Cohen's d effect size
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    # Remove NaN values
    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]

    if len(group1) < 2 or len(group2) < 2:
        return np.nan

    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return np.nan

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def get_significance_stars(p_value: float) -> str:
    """
    Convert p-value to significance stars.

    Parameters
    ----------
    p_value : float
        P-value from statistical test

    Returns
    -------
    str
        Significance annotation: '', '*', '**', or '***'
    """
    if np.isnan(p_value):
        return "ns"
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"


def add_stat_annotation(ax, x1: float, x2: float, y_max: float, p_value: float,
                       height_offset: float = 0.05):
    """
    Add significance annotation bracket to plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to annotate
    x1 : float
        First x-position
    x2 : float
        Second x-position
    y_max : float
        Y-position of bracket
    p_value : float
        P-value for significance
    height_offset : float
        Offset for bracket height
    """
    stars = get_significance_stars(p_value)
    if stars != "ns":
        y_pos = y_max * (1 + height_offset)
        ax.plot([x1, x1, x2, x2], [y_pos, y_pos * 1.01, y_pos * 1.01, y_pos], 'k-', lw=1.5)
        ax.text((x1 + x2) / 2, y_pos * 1.02, stars, ha='center', va='bottom', fontsize=11, fontweight='bold')


# ============================================================
# Main Comparison Class
# ============================================================

class ComparativeAnalysis:
    """
    Comprehensive multi-neuron comparative analysis class.

    Supports comparing 2 or more neurons across multiple analysis outputs.
    Includes statistical testing, effect size calculation, and publication-quality visualizations.

    Attributes
    ----------
    dirs : list of Path
        Directory paths for each neuron
    labels : list of str
        Labels for each neuron (e.g., "WT", "KO")
    data : dict
        Loaded analysis data organized by step
    output_dir : Path
        Output directory for results
    statistical_results : dict
        Dictionary storing all statistical test results
    """

    def __init__(self, dirs: List[Union[str, Path]], labels: Optional[List[str]] = None):
        """
        Initialize comparative analysis.

        Parameters
        ----------
        dirs : list of str or Path
            Directory paths for each neuron
        labels : list of str, optional
            Labels for each neuron. If None, uses "Neuron_1", "Neuron_2", etc.
        """
        self.dirs = [Path(d) for d in dirs]
        self.n_neurons = len(self.dirs)

        if labels is None:
            self.labels = [f"Neuron_{i+1}" for i in range(self.n_neurons)]
        else:
            if len(labels) != self.n_neurons:
                raise ValueError(f"Number of labels ({len(labels)}) must match number of directories ({self.n_neurons})")
            self.labels = labels

        self.data = {}
        self.output_dir = None
        self.statistical_results = {}

        print(f"✅ Initialized comparison for {self.n_neurons} neurons: {', '.join(self.labels)}")

    def set_output_directory(self, output_dir: Union[str, Path]):
        """Set and create output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory: {self.output_dir}")

    def load_all_data(self) -> None:
        """Load all available analysis data from input directories."""
        self.data = {}

        for step in range(1, 8):
            step_key = f"Step{step}"
            self.data[step_key] = {}

            csv_map = {
                1: "Step1_Lyso_Count_Outputs/Lysosome_Counts.csv",
                2: "Step2_Morphology_Outputs/Step2_Morphology_Summary.csv",
                3: "Step3_Shape_Analysis_Outputs/Step3_Mito_ShapeMetrics.csv",
                5: "Step5_Motility_Outputs/Step5_Motility_Summary.csv",
                6: "Step6_Colocalization_Outputs/Step6_Colocalization.csv",
                7: "Step7_Integrated_Summary_Outputs/Step7_Integrated_CorrelationMatrix.csv"
            }

            if step in csv_map:
                for i, dir_path in enumerate(self.dirs):
                    df = load_csv_safe(dir_path, csv_map[step])
                    if df is not None:
                        self.data[step_key][self.labels[i]] = df

    def violin_compare(self, metric: str, step_key: str, title: str) -> Tuple[plt.Figure, Optional[Dict]]:
        """
        Create violin plot comparing a metric across all neurons.

        Parameters
        ----------
        metric : str
            Column name to compare
        step_key : str
            Step identifier (e.g., "Step3")
        title : str
            Plot title

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        stats_dict : dict or None
            Statistical test results
        """
        # Prepare data
        dfs = []
        for label in self.labels:
            if label in self.data[step_key]:
                df = self.data[step_key][label].copy()
                df = df.reset_index(drop=True)
                df[metric] = pd.to_numeric(df[metric], errors='coerce')
                df = df.dropna(subset=[metric])
                df['Condition'] = label
                dfs.append(df)

        if not dfs:
            print(f"⚠️  No data for {metric} in {step_key}")
            return None, None

        combined = pd.concat(dfs, ignore_index=True)

        # Create figure
        fig, ax = plt.subplots(figsize=(max(6, self.n_neurons * 1.5), 5))

        # Violin plot with colorblind palette
        sns.violinplot(
            data=combined,
            x='Condition',
            y=metric,
            ax=ax,
            palette=CB_COLORS[:self.n_neurons],
            inner='quartile'
        )

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Condition', fontsize=11)
        ax.set_ylabel(metric, fontsize=11)

        # Statistical testing
        stats_dict = self._run_statistical_tests(combined, metric, step_key)

        # Add significance annotations for pairwise comparisons
        if self.n_neurons == 2 and stats_dict is not None and 'p_value' in stats_dict:
            y_max = combined[metric].max()
            add_stat_annotation(ax, 0, 1, y_max, stats_dict['p_value'])

        plt.tight_layout()
        return fig, stats_dict

    def _run_statistical_tests(self, combined_df: pd.DataFrame, metric: str,
                               step_key: str) -> Optional[Dict]:
        """
        Run appropriate statistical tests for the data.

        Parameters
        ----------
        combined_df : pd.DataFrame
            Combined data with 'Condition' column
        metric : str
            Metric column name
        step_key : str
            Step identifier

        Returns
        -------
        dict or None
            Dictionary containing test results
        """
        stats_dict = {}

        groups = [combined_df[combined_df['Condition'] == label][metric].values
                 for label in self.labels]

        # Remove empty groups
        groups = [g for g in groups if len(g) > 0]

        if len(groups) < 2:
            return None

        try:
            if self.n_neurons == 2:
                # Mann-Whitney U test (pairwise)
                stat, p_value = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                stats_dict['test'] = 'Mann-Whitney U'
                stats_dict['statistic'] = stat
                stats_dict['p_value'] = p_value
                stats_dict['effect_size'] = cohen_d(groups[0], groups[1])
            else:
                # Kruskal-Wallis test (multi-group)
                stat, p_value = kruskal(*groups)
                stats_dict['test'] = 'Kruskal-Wallis'
                stats_dict['statistic'] = stat
                stats_dict['p_value'] = p_value

                # Pairwise comparisons with Bonferroni correction
                n_comparisons = len(groups) * (len(groups) - 1) / 2
                bonferroni_alpha = 0.05 / n_comparisons

                pairwise_results = []
                for i in range(len(groups)):
                    for j in range(i + 1, len(groups)):
                        stat_ij, p_ij = mannwhitneyu(groups[i], groups[j], alternative='two-sided')
                        d_ij = cohen_d(groups[i], groups[j])
                        pairwise_results.append({
                            'comparison': f"{self.labels[i]} vs {self.labels[j]}",
                            'p_value': p_ij,
                            'p_value_bonferroni': p_ij * n_comparisons,
                            'significant_bonferroni': p_ij < bonferroni_alpha,
                            'effect_size': d_ij
                        })
                stats_dict['pairwise'] = pairwise_results
        except Exception as e:
            print(f"⚠️  Could not perform statistical test for {metric}: {e}")
            return None

        return stats_dict

    def compare_temporal_dynamics(self) -> Optional[pd.DataFrame]:
        """
        Compare temporal stability indices between neurons.

        Calculates coefficient of variation and trend analysis for time-series data.

        Returns
        -------
        pd.DataFrame or None
            Summary of temporal stability metrics
        """
        if "Step1" not in self.data:
            print("⚠️  Step 1 data not available for temporal analysis")
            return None

        results = []

        for label in self.labels:
            if label in self.data["Step1"]:
                df = self.data["Step1"][label]

                if 'Lysosome_Count' in df.columns:
                    lyso_counts = pd.to_numeric(df['Lysosome_Count'], errors='coerce').dropna()

                    # Coefficient of variation
                    cv = lyso_counts.std() / lyso_counts.mean() if lyso_counts.mean() > 0 else np.nan

                    # Trend (slope)
                    x = np.arange(len(lyso_counts))
                    if len(lyso_counts) > 1:
                        slope, _ = np.polyfit(x, lyso_counts.values, 1)
                    else:
                        slope = np.nan

                    results.append({
                        'Condition': label,
                        'Mean_Count': lyso_counts.mean(),
                        'Std_Dev': lyso_counts.std(),
                        'CV': cv,
                        'Trend_Slope': slope,
                        'N_Frames': len(lyso_counts)
                    })

        if results:
            temporal_df = pd.DataFrame(results)
            print("✅ Temporal dynamics analysis complete")
            return temporal_df

        return None

    def compare_spatial_statistics(self) -> Optional[pd.DataFrame]:
        """
        Compare spatial distributions between neurons.

        Analyzes spatial metrics like nearest neighbor distance (NND) and clustering.

        Returns
        -------
        pd.DataFrame or None
            Summary of spatial statistics
        """
        if "Step3" not in self.data:
            print("⚠️  Step 3 data not available for spatial analysis")
            return None

        results = []

        for label in self.labels:
            if label in self.data["Step3"]:
                df = self.data["Step3"][label]

                # Extract spatial metrics from Step 3
                spatial_metrics = {}
                for col in ['Area', 'Circularity', 'Solidity']:
                    if col in df.columns:
                        values = pd.to_numeric(df[col], errors='coerce').dropna()
                        if len(values) > 0:
                            spatial_metrics[col] = {
                                'mean': values.mean(),
                                'std': values.std(),
                                'median': values.median()
                            }

                results.append({
                    'Condition': label,
                    'N_Objects': len(df),
                    **{f"{k}_mean": v['mean'] for k, v in spatial_metrics.items()},
                    **{f"{k}_std": v['std'] for k, v in spatial_metrics.items()}
                })

        if results:
            spatial_df = pd.DataFrame(results)
            print("✅ Spatial statistics analysis complete")
            return spatial_df

        return None

    def compare_network_topology(self) -> Optional[pd.DataFrame]:
        """
        Compare network topology metrics between neurons.

        Analyzes interconnectedness and fragmentation patterns.

        Returns
        -------
        pd.DataFrame or None
            Summary of network topology metrics
        """
        # This function would integrate with Step 7 (integrated analysis)
        # and any network analysis data available

        if "Step7" not in self.data:
            print("⚠️  Step 7 data not available for network analysis")
            return None

        results = []

        for label in self.labels:
            if label in self.data["Step7"]:
                df = self.data["Step7"][label]

                # Calculate correlation-based network metrics
                # Correlation matrix density as a proxy for connectivity
                numeric_cols = df.select_dtypes(include=[np.number]).columns

                if len(numeric_cols) > 1:
                    corr_matrix = df[numeric_cols].corr()
                    # Network density: proportion of non-zero correlations
                    density = (np.abs(corr_matrix.values) > 0.3).sum() / (len(corr_matrix) ** 2 - len(corr_matrix))

                    results.append({
                        'Condition': label,
                        'Network_Density': density,
                        'N_Features': len(numeric_cols),
                        'Mean_Correlation': corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
                    })

        if results:
            network_df = pd.DataFrame(results)
            print("✅ Network topology analysis complete")
            return network_df

        return None

    def generate_comparison_report(self) -> None:
        """
        Generate comprehensive comparison report with statistics and visualizations.

        Creates:
        - Comparison report CSV with all metrics and statistics
        - Multi-panel figure with key comparisons
        """
        if self.output_dir is None:
            raise ValueError("Output directory not set. Call set_output_directory() first.")

        report_data = []

        # Summary statistics for each step
        steps_to_analyze = {
            'Step3': ['Circularity', 'Solidity', 'Aspect_Ratio', 'Eccentricity', 'Area'],
            'Step5': ['Mean_Velocity', 'Total_Displacement'],
            'Step6': ['Manders_M1', 'Manders_M2', 'Pearson_r', 'Percent_Overlap']
        }

        for step_key, metrics in steps_to_analyze.items():
            if step_key not in self.data:
                continue

            for metric in metrics:
                stat_results = self._get_or_compute_stats(step_key, metric)

                for label in self.labels:
                    if label in self.data[step_key]:
                        df = self.data[step_key][label]
                        if metric in df.columns:
                            values = pd.to_numeric(df[metric], errors='coerce').dropna()

                            report_entry = {
                                'Step': step_key,
                                'Metric': metric,
                                'Condition': label,
                                'N': len(values),
                                'Mean': values.mean(),
                                'Median': values.median(),
                                'Std': values.std(),
                                'Min': values.min(),
                                'Max': values.max(),
                            }

                            if stat_results and 'pairwise' in stat_results:
                                for pw in stat_results['pairwise']:
                                    if label in pw['comparison']:
                                        report_entry['P_Value'] = pw['p_value']
                                        report_entry['Effect_Size_Cohen_d'] = pw['effect_size']
                                        break

                            report_data.append(report_entry)

        if report_data:
            report_df = pd.DataFrame(report_data)
            report_path = self.output_dir / "Comparison_Report.csv"
            report_df.to_csv(report_path, index=False)
            print(f"📊 Comparison report saved: {report_path}")

        # Temporal dynamics report
        temporal_df = self.compare_temporal_dynamics()
        if temporal_df is not None:
            temporal_path = self.output_dir / "Temporal_Dynamics_Summary.csv"
            temporal_df.to_csv(temporal_path, index=False)
            print(f"⏱️  Temporal dynamics report saved: {temporal_path}")

        # Spatial statistics report
        spatial_df = self.compare_spatial_statistics()
        if spatial_df is not None:
            spatial_path = self.output_dir / "Spatial_Statistics_Summary.csv"
            spatial_df.to_csv(spatial_path, index=False)
            print(f"📍 Spatial statistics report saved: {spatial_path}")

        # Network topology report
        network_df = self.compare_network_topology()
        if network_df is not None:
            network_path = self.output_dir / "Network_Topology_Summary.csv"
            network_df.to_csv(network_path, index=False)
            print(f"🔗 Network topology report saved: {network_path}")

    def _get_or_compute_stats(self, step_key: str, metric: str) -> Optional[Dict]:
        """Retrieve or compute statistics for a metric."""
        key = f"{step_key}_{metric}"
        if key not in self.statistical_results:
            dfs = []
            for label in self.labels:
                if label in self.data.get(step_key, {}):
                    df = self.data[step_key][label].copy()
                    df[metric] = pd.to_numeric(df[metric], errors='coerce')
                    df = df.dropna(subset=[metric])
                    df['Condition'] = label
                    dfs.append(df)

            if dfs:
                combined = pd.concat(dfs, ignore_index=True)
                self.statistical_results[key] = self._run_statistical_tests(combined, metric, step_key)

        return self.statistical_results.get(key)

    def run_full_analysis(self, output_dir: Union[str, Path]) -> None:
        """
        Run complete comparative analysis pipeline.

        Parameters
        ----------
        output_dir : str or Path
            Output directory for results
        """
        self.set_output_directory(output_dir)
        self.load_all_data()

        print("\n" + "="*60)
        print("Running Comparative Analysis Pipeline")
        print("="*60 + "\n")

        # Step 1: Lysosome count comparison
        if "Step1" in self.data:
            self._compare_step1_lyso_count()

        # Step 2: Morphology comparison
        if "Step2" in self.data:
            self._compare_step2_morphology()

        # Step 3: Shape metrics
        if "Step3" in self.data:
            self._compare_step3_shape()

        # Step 5: Motility
        if "Step5" in self.data:
            self._compare_step5_motility()

        # Step 6: Colocalization
        if "Step6" in self.data:
            self._compare_step6_colocalization()

        # Step 7: Integrated correlation
        if "Step7" in self.data:
            self._compare_step7_integrated()

        # Generate comprehensive report
        self.generate_comparison_report()

        print("\n" + "="*60)
        print(f"✅ Analysis complete! Results saved to: {self.output_dir}")
        print("="*60 + "\n")

    def _compare_step1_lyso_count(self) -> None:
        """Compare lysosome counts per frame."""
        fig, ax = plt.subplots(figsize=(8, 5))

        for i, label in enumerate(self.labels):
            if label in self.data["Step1"]:
                df = self.data["Step1"][label]
                if 'Frame' in df.columns and 'Lysosome_Count' in df.columns:
                    ax.plot(df['Frame'], df['Lysosome_Count'], '-o',
                           label=label, color=CB_COLORS[i], linewidth=2)

        ax.set_xlabel('Frame', fontsize=11)
        ax.set_ylabel('Lysosome Count', fontsize=11)
        ax.set_title('Lysosome Count Comparison per Frame', fontsize=12, fontweight='bold')
        ax.legend()
        plt.tight_layout()

        out_path = self.output_dir / "Step1_LysoCount_Comparison.png"
        plt.savefig(out_path, dpi=600)
        plt.close()
        print(f"✅ Step 1 comparison saved: {out_path.name}")

    def _compare_step2_morphology(self) -> None:
        """Compare morphology (elongated vs punctate)."""
        fig, ax = plt.subplots(figsize=(10, 5))

        dfs = []
        for label in self.labels:
            if label in self.data["Step2"]:
                df = self.data["Step2"][label].copy()
                df['Condition'] = label
                dfs.append(df)

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            melted = combined.melt(id_vars=['Frame', 'Condition'],
                                  value_vars=['Elongated', 'Punctate'],
                                  var_name='Type', value_name='Count')

            sns.lineplot(data=melted, x='Frame', y='Count', hue='Type',
                        style='Condition', markers=True, ax=ax, palette='Set2')
            ax.set_title('Mitochondrial Morphology Comparison', fontsize=12, fontweight='bold')
            plt.tight_layout()

            out_path = self.output_dir / "Step2_Morphology_Comparison.png"
            plt.savefig(out_path, dpi=600)
            plt.close()
            print(f"✅ Step 2 comparison saved: {out_path.name}")

    def _compare_step3_shape(self) -> None:
        """Compare shape metrics."""
        metrics = ['Circularity', 'Solidity', 'Aspect_Ratio', 'Eccentricity', 'Area']

        for metric in metrics:
            fig, stats_dict = self.violin_compare(metric, 'Step3', f'Step 3: {metric}')

            if fig is not None:
                out_path = self.output_dir / f"Step3_{metric}_Comparison.png"
                plt.savefig(out_path, dpi=600)
                plt.close()
                print(f"✅ Step 3 - {metric} saved: {out_path.name}")

        print("✅ Step 3 shape metrics comparison complete.")

    def _compare_step5_motility(self) -> None:
        """Compare motility metrics."""
        metrics = ['Mean_Velocity', 'Total_Displacement']

        for metric in metrics:
            fig, stats_dict = self.violin_compare(metric, 'Step5', f'Step 5: {metric}')

            if fig is not None:
                out_path = self.output_dir / f"Step5_{metric}_Comparison.png"
                plt.savefig(out_path, dpi=600)
                plt.close()
                print(f"✅ Step 5 - {metric} saved: {out_path.name}")

        print("✅ Step 5 motility comparison complete.")

    def _compare_step6_colocalization(self) -> None:
        """Compare colocalization metrics."""
        metrics = ['Manders_M1', 'Manders_M2', 'Pearson_r', 'Percent_Overlap']

        for metric in metrics:
            fig, stats_dict = self.violin_compare(metric, 'Step6', f'Step 6: {metric}')

            if fig is not None:
                out_path = self.output_dir / f"Step6_{metric}_Comparison.png"
                plt.savefig(out_path, dpi=600)
                plt.close()
                print(f"✅ Step 6 - {metric} saved: {out_path.name}")

        print("✅ Step 6 colocalization comparison complete.")

    def _compare_step7_integrated(self) -> None:
        """Compare integrated correlation matrices."""
        if self.n_neurons == 2:
            label1, label2 = self.labels[0], self.labels[1]

            if label1 in self.data["Step7"] and label2 in self.data["Step7"]:
                df1 = self.data["Step7"][label1]
                df2 = self.data["Step7"][label2]

                diff = df1.set_index(df1.columns[0]).subtract(
                    df2.set_index(df2.columns[0]), fill_value=0
                )

                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(diff, cmap='coolwarm', center=0, ax=ax, cbar_kws={'label': 'Correlation Difference'})
                ax.set_title(f'Step 7: Correlation Difference ({label1} – {label2})',
                            fontsize=12, fontweight='bold')
                plt.tight_layout()

                out_path = self.output_dir / "Step7_CorrelationMatrix_Difference.png"
                plt.savefig(out_path, dpi=600)
                plt.close()
                print(f"✅ Step 7 comparison saved: {out_path.name}")


# ============================================================
# Command-line Interface
# ============================================================

def main():
    """Command-line interface for AMTComparison."""
    parser = argparse.ArgumentParser(
        description='Advanced Multi-Neuron Analysis Comparison Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python AMTComparison.py --dirs neuron1/ neuron2/ --labels "WT" "KO" --output results/
  python AMTComparison.py --dirs dir1 dir2 dir3 --output comparison/
  python AMTComparison.py -d n1 n2 n3 -l A B C -o output/
        """
    )

    parser.add_argument('--dirs', '-d', nargs='+', required=True,
                       help='Input directories containing analysis outputs')
    parser.add_argument('--labels', '-l', nargs='+', default=None,
                       help='Labels for each neuron (e.g., WT KO Rescue)')
    parser.add_argument('--output', '-o', default='Comparison_Results',
                       help='Output directory for results (default: Comparison_Results)')

    args = parser.parse_args()

    # Run analysis
    try:
        ca = ComparativeAnalysis(dirs=args.dirs, labels=args.labels)
        ca.run_full_analysis(output_dir=args.output)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
