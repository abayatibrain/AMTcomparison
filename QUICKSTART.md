# Quick Start Guide - AMTComparison 2.0.0

Get started with AMTComparison in 5 minutes!

## Installation

```bash
# Navigate to the project directory
cd /tmp/amt-comparison-work

# Install dependencies
pip install numpy pandas matplotlib seaborn scipy

# Install AMTComparison in development mode
pip install -e .
```

## Basic Usage

### Command-Line (Easiest)

```bash
# Compare two neurons
python AMTComparison.py --dirs neuron1_output neuron2_output --output results

# Compare with custom labels
python AMTComparison.py -d wt ko -l "Wild-Type" "Knockout" -o wt_vs_ko

# Compare multiple groups
python AMTComparison.py --dirs wt_output ko_output rescue_output \
  --labels "WT" "KO" "Rescue" --output three_group_analysis
```

### Python Script

```python
from AMTComparison import ComparativeAnalysis

# Initialize
ca = ComparativeAnalysis(
    dirs=['neuron1/', 'neuron2/'],
    labels=['Control', 'Treatment']
)

# Run analysis
ca.run_full_analysis(output_dir='my_results/')

# Check results
print("Analysis complete! Check my_results/ for:")
print("  - Comparison_Report.csv")
print("  - PNG figures (600 DPI)")
print("  - Summary statistics")
```

## Expected Input Structure

Your input directories should look like this:

```
neuron_output/
├── Step1_Lyso_Count_Outputs/
│   └── Lysosome_Counts.csv
├── Step3_Shape_Analysis_Outputs/
│   └── Step3_Mito_ShapeMetrics.csv
├── Step5_Motility_Outputs/
│   └── Step5_Motility_Summary.csv
├── Step6_Colocalization_Outputs/
│   └── Step6_Colocalization.csv
└── (other analysis folders...)
```

Not all steps are required - the tool gracefully handles missing data.

## Output Files

After running, you'll get:

### Reports
- **Comparison_Report.csv** - All statistics in tabular format
  - Mean, Std, P-value, Effect size for each metric
- **Temporal_Dynamics_Summary.csv** - Time-series stability
- **Spatial_Statistics_Summary.csv** - Spatial patterns
- **Network_Topology_Summary.csv** - Connectivity metrics

### Figures (Publication-Ready)
- **Step1_LysoCount_Comparison.png**
- **Step3_Area_Comparison.png**, **Step3_Circularity_Comparison.png**, etc.
- **Step5_Mean_Velocity_Comparison.png**, **Step5_Total_Displacement_Comparison.png**
- **Step6_Manders_M1_Comparison.png**, **Step6_Pearson_r_Comparison.png**, etc.
- **Step7_CorrelationMatrix_Difference.png**

All figures are 600 DPI with significance annotations (* p<0.05, ** p<0.01, *** p<0.001).

## Understanding Results

### Key Metrics in CSV Reports

| Column | Meaning |
|--------|---------|
| **Mean** | Average value for the group |
| **Std** | Standard deviation (variability) |
| **P_Value** | Statistical significance (lower = more different) |
| **Effect_Size_Cohen_d** | Magnitude of difference (|d|: 0.2=small, 0.5=med, 0.8=large) |

### P-Value Interpretation

```
p < 0.001  →  ***  (Highly significant)
p < 0.01   →  **   (Very significant)
p < 0.05   →  *    (Significant)
p ≥ 0.05   →  ns   (Not significant)
```

### Effect Size (Cohen's d)

```
|d| < 0.2       →  Negligible effect
0.2 ≤ |d| < 0.5 →  Small effect
0.5 ≤ |d| < 0.8 →  Medium effect
|d| ≥ 0.8       →  Large effect
```

## Common Tasks

### See all available options
```bash
python AMTComparison.py --help
```

### Compare 3+ groups with automatic Bonferroni correction
```bash
python AMTComparison.py -d group1 group2 group3 group4 \
  -l "A" "B" "C" "D" -o multi_group_analysis
```

### Access specific statistical results in Python
```python
ca = ComparativeAnalysis(dirs=['d1', 'd2'], labels=['G1', 'G2'])
ca.set_output_directory('output/')
ca.load_all_data()

# Get stats for a specific metric
stats = ca._get_or_compute_stats('Step3', 'Area')
print(f"P-value: {stats['p_value']}")
print(f"Effect size: {stats['effect_size']}")
```

### Generate just temporal analysis
```python
ca = ComparativeAnalysis(dirs=['d1', 'd2'], labels=['G1', 'G2'])
ca.set_output_directory('temp_analysis/')
ca.load_all_data()
temporal_df = ca.compare_temporal_dynamics()
print(temporal_df)
```

## Troubleshooting

### "Missing file" warnings
- Not all analysis steps may be available for all neurons
- Tool automatically skips missing data
- Only affects the missing step; other analyses continue

### Figures not saving
- Check output directory exists and is writable
- Ensure disk space is available
- Try specifying full path: `-o /path/to/output`

### Statistical tests fail
- Requires at least 2 samples per group
- Check that CSV files contain numeric data
- Ensure columns match expected names (Area, Circularity, etc.)

### Need help?
- See **README.md** for full documentation
- See **EXAMPLES.md** for detailed use cases
- Check **IMPLEMENTATION_DETAILS.md** for technical info

## Next Steps

1. **Run your first comparison**
   ```bash
   python AMTComparison.py -d neuron1 neuron2 -o first_test
   ```

2. **Review the results**
   ```bash
   cat first_test/Comparison_Report.csv
   open first_test/*.png  # or use your image viewer
   ```

3. **Customize for your needs**
   - Add custom labels with `-l`
   - Change output directory with `-o`
   - Process multiple groups automatically

4. **Integrate into workflows**
   - Use `from AMTComparison import ComparativeAnalysis` in your scripts
   - Chain with other Python data analysis tools
   - Automate batch processing with shell scripts

## Example Workflow Script

```python
#!/usr/bin/env python
"""
Automated batch comparison of multiple neuron pairs
"""
from pathlib import Path
from AMTComparison import ComparativeAnalysis
import pandas as pd

# Define your comparisons
comparisons = [
    (['WT1', 'KO1'], ['WT', 'KO'], 'wt_vs_ko_pair1'),
    (['WT2', 'KO2'], ['WT', 'KO'], 'wt_vs_ko_pair2'),
    (['Rescue1', 'KO1'], ['Rescue', 'KO'], 'rescue_vs_ko'),
]

# Run all comparisons
all_results = []
for dirs, labels, output_name in comparisons:
    print(f"\nProcessing: {output_name}")
    ca = ComparativeAnalysis(dirs=dirs, labels=labels)
    ca.run_full_analysis(output_dir=output_name)

    # Load results
    report = pd.read_csv(f'{output_name}/Comparison_Report.csv')
    all_results.append(report)

# Combine all results
combined = pd.concat(all_results, ignore_index=True)
combined.to_csv('all_comparisons_summary.csv', index=False)
print("\nAll comparisons saved to all_comparisons_summary.csv")
```

Save as `batch_compare.py` and run:
```bash
python batch_compare.py
```

---

**Happy analyzing!** 🧬📊

For detailed documentation, see README.md and EXAMPLES.md
