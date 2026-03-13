# AMTComparison Examples and Use Cases

This document provides detailed examples for using AMTComparison in various scenarios.

## Table of Contents
1. [Command-line Examples](#command-line-examples)
2. [Python API Examples](#python-api-examples)
3. [Advanced Analysis Examples](#advanced-analysis-examples)
4. [Output Interpretation Guide](#output-interpretation-guide)

---

## Command-line Examples

### Example 1: Basic Two-Neuron Comparison

Compare two neurons with default naming:
```bash
python AMTComparison.py --dirs neuron1_output neuron2_output --output results
```

This will:
- Load all analysis data from both directories
- Compare all available metrics
- Generate CSV reports with statistics
- Create publication-quality figures (600 DPI)
- Name neurons "Neuron_1" and "Neuron_2"

### Example 2: Two-Group Comparison with Labels

Compare wild-type vs knockout neurons:
```bash
python AMTComparison.py \
  --dirs WT_neurons KO_neurons \
  --labels "WT" "KO" \
  --output WT_vs_KO_analysis
```

### Example 3: Multi-Group Comparison (3+ Groups)

Compare multiple genotypes:
```bash
python AMTComparison.py \
  --dirs wt_outputs ko_outputs rescue_outputs \
  --labels "Wild-Type" "Knockout" "Rescue" \
  --output three_group_comparison
```

This will:
- Run Kruskal-Wallis tests (instead of Mann-Whitney U)
- Calculate pairwise comparisons with Bonferroni correction
- Generate separate p-values for each pair
- Create effect size (Cohen's d) for all comparisons

### Example 4: Using Short-form Arguments

```bash
python AMTComparison.py -d dir1 dir2 dir3 -l A B C -o output
```

---

## Python API Examples

### Example 1: Basic Comparative Analysis

```python
from AMTComparison import ComparativeAnalysis

# Initialize with two neurons
ca = ComparativeAnalysis(
    dirs=['neuron1_output/', 'neuron2_output/'],
    labels=['Control', 'Treatment']
)

# Run full analysis
ca.run_full_analysis(output_dir='results/')
```

### Example 2: Step-by-Step Analysis Control

```python
from AMTComparison import ComparativeAnalysis

# Initialize
ca = ComparativeAnalysis(
    dirs=['sample1/', 'sample2/', 'sample3/'],
    labels=['WT', 'KO', 'Rescue']
)

# Set output directory
ca.set_output_directory('custom_output/')

# Load data
ca.load_all_data()

# Run specific analyses
print("Loading temporal dynamics...")
temporal_data = ca.compare_temporal_dynamics()
print(temporal_data)

print("\nLoading spatial statistics...")
spatial_data = ca.compare_spatial_statistics()
print(spatial_data)

print("\nLoading network topology...")
network_data = ca.compare_network_topology()
print(network_data)

# Generate comprehensive report
ca.generate_comparison_report()
```

### Example 3: Create Specific Visualizations with Statistics

```python
from AMTComparison import ComparativeAnalysis

ca = ComparativeAnalysis(
    dirs=['control/', 'treatment/'],
    labels=['Control', 'Treatment']
)
ca.set_output_directory('visualizations/')
ca.load_all_data()

# Create violin plot for specific metric
fig, stats = ca.violin_compare(
    metric='Area',
    step_key='Step3',
    title='Mitochondrial Area Comparison'
)

# Print statistics
if stats:
    print(f"Statistical Test: {stats['test']}")
    print(f"P-value: {stats['p_value']:.4f}")
    print(f"Effect Size (Cohen's d): {stats['effect_size']:.3f}")

# Save figure
fig.savefig('visualizations/area_comparison_with_stats.png', dpi=600, bbox_inches='tight')
```

### Example 4: Access Statistical Results

```python
from AMTComparison import ComparativeAnalysis

ca = ComparativeAnalysis(
    dirs=['group1/', 'group2/', 'group3/'],
    labels=['Group A', 'Group B', 'Group C']
)
ca.set_output_directory('stats_output/')
ca.load_all_data()

# Get statistics for specific metrics
metrics_to_check = ['Area', 'Circularity', 'Solidity']

for metric in metrics_to_check:
    stats = ca._get_or_compute_stats('Step3', metric)
    if stats:
        print(f"\n{metric}:")
        print(f"  Test: {stats['test']}")
        print(f"  P-value: {stats['p_value']:.4f}")

        if 'pairwise' in stats:
            print("  Pairwise comparisons:")
            for pw in stats['pairwise']:
                print(f"    {pw['comparison']}: p={pw['p_value']:.4f}, d={pw['effect_size']:.3f}")
```

### Example 5: Custom Batch Analysis

```python
from AMTComparison import ComparativeAnalysis
import glob

# Find all neuron directories matching pattern
neuron_dirs = sorted(glob.glob('neurons/neuron_*'))
labels = [f"N{i}" for i in range(1, len(neuron_dirs) + 1)]

# Run analysis
ca = ComparativeAnalysis(dirs=neuron_dirs, labels=labels)
ca.run_full_analysis(output_dir='batch_comparison')

# Read and display summary report
import pandas as pd
report = pd.read_csv('batch_comparison/Comparison_Report.csv')

# Show significant differences
significant = report[report['P_Value'] < 0.05]
print("\nSignificant differences (p < 0.05):")
print(significant[['Step', 'Metric', 'Condition', 'Mean', 'P_Value']])
```

---

## Advanced Analysis Examples

### Example 1: Selective Data Analysis (Skip Missing Steps)

```python
from AMTComparison import ComparativeAnalysis

ca = ComparativeAnalysis(dirs=['dir1', 'dir2'], labels=['A', 'B'])
ca.set_output_directory('results/')
ca.load_all_data()

# Only analyze steps that have data
available_steps = list(ca.data.keys())
print(f"Available analysis steps: {available_steps}")

# Manually run only available analyses
for step in available_steps:
    print(f"Processing {step}...")
    # Custom processing for each step
```

### Example 2: Effect Size Interpretation

```python
from AMTComparison import ComparativeAnalysis
import pandas as pd

ca = ComparativeAnalysis(dirs=['sample1/', 'sample2/'], labels=['S1', 'S2'])
ca.run_full_analysis('effect_size_analysis/')

# Read report and categorize effect sizes
report = pd.read_csv('effect_size_analysis/Comparison_Report.csv')

# Add effect size interpretation
def interpret_cohens_d(d):
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"

report['Effect_Interpretation'] = report['Effect_Size_Cohen_d'].apply(interpret_cohens_d)

print("\nEffect Size Summary:")
print(report.groupby('Effect_Interpretation').size())
print("\nDetailed results:")
print(report[['Metric', 'Condition', 'Effect_Size_Cohen_d', 'Effect_Interpretation', 'P_Value']])
```

### Example 3: Temporal Trend Analysis

```python
from AMTComparison import ComparativeAnalysis
import pandas as pd
import matplotlib.pyplot as plt

ca = ComparativeAnalysis(dirs=['control/', 'disease/'], labels=['Control', 'Disease'])
ca.set_output_directory('temporal_analysis/')
ca.load_all_data()

# Get temporal dynamics
temporal = ca.compare_temporal_dynamics()
print("Temporal Dynamics Summary:")
print(temporal)

# Visualize temporal stability
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Coefficient of Variation
temporal.plot(x='Condition', y='CV', kind='bar', ax=ax1)
ax1.set_title('Temporal Stability (Coefficient of Variation)')
ax1.set_ylabel('CV')
ax1.set_xlabel('Group')

# Plot 2: Trend slopes
temporal.plot(x='Condition', y='Trend_Slope', kind='bar', ax=ax2)
ax2.set_title('Temporal Trend (Slope)')
ax2.set_ylabel('Slope')
ax2.set_xlabel('Group')

plt.tight_layout()
plt.savefig('temporal_analysis/temporal_summary.png', dpi=600)
```

### Example 4: Multi-level Statistical Summary

```python
from AMTComparison import ComparativeAnalysis
import pandas as pd

ca = ComparativeAnalysis(
    dirs=['wt/', 'het/', 'ko/'],
    labels=['WT', 'Het', 'KO']
)
ca.run_full_analysis('statistical_summary/')

# Load report
report = pd.read_csv('statistical_summary/Comparison_Report.csv')

# Group by metric and show statistics
for metric in report['Metric'].unique():
    metric_data = report[report['Metric'] == metric]
    print(f"\n{'='*60}")
    print(f"Metric: {metric}")
    print(f"{'='*60}")

    # Summary statistics
    for _, row in metric_data.iterrows():
        print(f"{row['Condition']}: Mean={row['Mean']:.3f} ± {row['Std']:.3f} (n={int(row['N'])})")

    # Statistical test result
    p_val = metric_data['P_Value'].iloc[0]
    print(f"P-value: {p_val:.4f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else '(ns)'}")
```

---

## Output Interpretation Guide

### Understanding CSV Reports

#### Comparison_Report.csv Columns:
- **Step**: Analysis step (Step1, Step3, etc.)
- **Metric**: Measurement name (Area, Circularity, etc.)
- **Condition**: Group/Neuron label
- **N**: Sample size
- **Mean**: Average value
- **Median**: Median value
- **Std**: Standard deviation
- **Min/Max**: Range of values
- **P_Value**: Statistical significance (lower = more significant)
- **Effect_Size_Cohen_d**: Magnitude of difference

#### Interpreting P-values:
```
p < 0.001  →  *** (Highly significant - very strong evidence)
p < 0.01   →  ** (Very significant - strong evidence)
p < 0.05   →  * (Significant - evidence of difference)
p ≥ 0.05   →  ns (Not significant - no clear evidence)
```

#### Interpreting Cohen's d:
```
|d| < 0.2      →  Negligible effect size
0.2 ≤ |d| < 0.5 →  Small effect size
0.5 ≤ |d| < 0.8 →  Medium effect size
|d| ≥ 0.8       →  Large effect size
```

### Understanding Figures

#### Violin Plots:
- **Width**: Distribution density at each value
- **White dot**: Median
- **Thick black bar**: Interquartile range (25-75%)
- **Whiskers**: Min-max range
- **Significance stars**: *, **, *** indicate p < 0.05, 0.01, 0.001

#### Line Plots:
- Show temporal trends in metrics
- Each line represents one condition/group
- Useful for identifying temporal patterns

#### Heatmaps:
- Show correlation matrix differences
- Red = positive correlation difference
- Blue = negative correlation difference
- Intensity = magnitude of difference

### Example Report Reading

```
Step: Step3
Metric: Area
Condition: Control, Mean: 450.5 ± 89.3, N: 245, P_Value: 0.0023, Cohen's d: 0.63
Condition: Treatment, Mean: 389.2 ± 101.4, N: 238

Interpretation:
- Treatment neurons have significantly smaller mitochondrial areas
- This is a medium effect size (d = 0.63)
- The difference is highly significant (p = 0.0023)
- Both groups have similar variability (SD: ~95)
```

---

## Performance Tips

1. **Large Datasets**: For very large neuron datasets, analysis may take several minutes
2. **Memory Usage**: Keep intermediate results by using step-by-step analysis
3. **Batch Processing**: Use glob patterns to process multiple neurons
4. **Parallel Analysis**: Run separate comparisons on different CPU cores

---

## Troubleshooting

### Missing Data Warnings
```
⚠️  Missing file: /path/to/Step1_Lyso_Count_Outputs/Lysosome_Counts.csv
```
- Not all analysis steps may be available for all neurons
- Tool automatically skips missing data
- Check input directory structure matches expected format

### Statistical Test Limitations
- Mann-Whitney U test requires at least 2 samples per group
- Kruskal-Wallis requires at least 3 samples total
- NaN values are automatically removed during analysis

### Visualization Issues
- Ensure matplotlib backend is properly configured
- For headless servers, use `matplotlib.use('Agg')`
- Check output directory has write permissions
