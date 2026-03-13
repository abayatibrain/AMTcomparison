# AMTComparison: Advanced Multi-Neuron Analysis Comparison Tool

A comprehensive Python toolkit for comparing multiple neurons' analysis outputs from the Advanced Morphology Tool (AMT) pipeline. Supports statistical testing, effect size calculation, and publication-quality visualizations for multi-group neuron comparisons.

## Features

### Core Capabilities
- **Multi-neuron comparison**: Compare 2 or more neurons simultaneously
- **Statistical testing**:
  - Mann-Whitney U test for pairwise comparisons
  - Kruskal-Wallis test for multi-group comparisons
  - Cohen's d effect size calculation
  - Bonferroni correction for multiple comparisons
- **Comprehensive metrics analysis**:
  - Lysosome count dynamics
  - Mitochondrial morphology (elongated vs punctate)
  - Shape analysis (circularity, solidity, aspect ratio, eccentricity)
  - Motility metrics (velocity, displacement)
  - Colocalization analysis (Manders, Pearson, overlap)
  - Integrated correlation matrices
  - Temporal dynamics and stability
  - Spatial statistics
  - Network topology

### Visualizations
- **Colorblind-safe palette**: Uses CB-friendly colors for accessibility
- **Publication-quality figures**: 600 DPI output with proper scaling
- **Significance annotations**: Statistical markers (* p<0.05, ** p<0.01, *** p<0.001)
- **Effect size annotations**: Shows Cohen's d values
- **Violin plots**: Distribution comparisons with quartile indicators
- **Line plots**: Temporal dynamics visualization
- **Heatmaps**: Correlation matrix comparisons

### Reporting
- Comprehensive CSV report with all metrics and statistics
- Temporal dynamics summary
- Spatial statistics summary
- Network topology summary
- Multi-panel figures for easy publication

## Installation

### Via pip (after publishing)
```bash
pip install AMTComparison
```

### From source
```bash
git clone https://github.com/abayatibrain/AMTcomparison.git
cd AMTcomparison
pip install -e .
```

### Requirements
- Python >= 3.7
- numpy >= 1.19.0
- pandas >= 1.1.0
- matplotlib >= 3.2.0
- seaborn >= 0.11.0
- scipy >= 1.5.0

## Usage

### Command-line Interface

Basic usage with two neurons:
```bash
python AMTComparison.py --dirs neuron1_outputs/ neuron2_outputs/ --output results/
```

Multi-neuron comparison with labels:
```bash
python AMTComparison.py \
  --dirs wt_neuron/ ko_neuron/ rescue_neuron/ \
  --labels "WT" "KO" "Rescue" \
  --output comparison_results/
```

Short form:
```bash
python AMTComparison.py -d dir1 dir2 dir3 -l A B C -o output/
```

### Python API

#### Basic Usage
```python
from AMTComparison import ComparativeAnalysis

# Initialize with two neurons
ca = ComparativeAnalysis(
    dirs=['neuron1_outputs/', 'neuron2_outputs/'],
    labels=['WT', 'KO']
)

# Run full analysis pipeline
ca.run_full_analysis(output_dir='results/')
```

#### Advanced Usage with Custom Analysis
```python
from AMTComparison import ComparativeAnalysis

# Initialize with multiple neurons
ca = ComparativeAnalysis(
    dirs=['wt/', 'ko/', 'rescue/'],
    labels=['WT', 'KO', 'Rescue']
)

# Set output directory
ca.set_output_directory('comparison_output/')

# Load all data
ca.load_all_data()

# Perform specific comparisons
temporal = ca.compare_temporal_dynamics()
spatial = ca.compare_spatial_statistics()
network = ca.compare_network_topology()

# Generate reports
ca.generate_comparison_report()

# Access specific metrics
fig, stats = ca.violin_compare(
    metric='Area',
    step_key='Step3',
    title='Mitochondrial Area Comparison'
)
```

#### Accessing Statistical Results
```python
from AMTComparison import ComparativeAnalysis

ca = ComparativeAnalysis(dirs=['dir1', 'dir2'], labels=['Group1', 'Group2'])
ca.set_output_directory('output/')
ca.load_all_data()

# Get statistics for a specific metric
stats = ca._get_or_compute_stats('Step3', 'Area')
print(f"P-value: {stats['p_value']}")
print(f"Effect size (Cohen's d): {stats['effect_size']}")
```

## Input Data Structure

The input directories should follow the standard AMT output structure:

```
neuron_outputs/
├── Step1_Lyso_Count_Outputs/
│   └── Lysosome_Counts.csv
├── Step2_Morphology_Outputs/
│   └── Step2_Morphology_Summary.csv
├── Step3_Shape_Analysis_Outputs/
│   └── Step3_Mito_ShapeMetrics.csv
├── Step5_Motility_Outputs/
│   └── Step5_Motility_Summary.csv
├── Step6_Colocalization_Outputs/
│   └── Step6_Colocalization.csv
└── Step7_Integrated_Summary_Outputs/
    └── Step7_Integrated_CorrelationMatrix.csv
```

## Output Files

### Reports
- **Comparison_Report.csv**: Comprehensive statistics for all metrics
  - Columns: Step, Metric, Condition, N, Mean, Median, Std, Min, Max, P_Value, Effect_Size_Cohen_d
- **Temporal_Dynamics_Summary.csv**: Time-series stability metrics
  - Mean_Count, Std_Dev, CV (coefficient of variation), Trend_Slope
- **Spatial_Statistics_Summary.csv**: Spatial distribution metrics
  - Area_mean, Circularity_mean, Solidity_mean, and standard deviations
- **Network_Topology_Summary.csv**: Network connectivity metrics
  - Network_Density, N_Features, Mean_Correlation

### Figures (600 DPI PNG)
#### Step 1 - Lysosome Dynamics
- `Step1_LysoCount_Comparison.png`: Lysosome count over frames

#### Step 2 - Morphology
- `Step2_Morphology_Comparison.png`: Elongated vs punctate distribution

#### Step 3 - Shape Metrics
- `Step3_Area_Comparison.png`
- `Step3_Circularity_Comparison.png`
- `Step3_Solidity_Comparison.png`
- `Step3_Aspect_Ratio_Comparison.png`
- `Step3_Eccentricity_Comparison.png`

#### Step 5 - Motility
- `Step5_Mean_Velocity_Comparison.png`
- `Step5_Total_Displacement_Comparison.png`

#### Step 6 - Colocalization
- `Step6_Manders_M1_Comparison.png`
- `Step6_Manders_M2_Comparison.png`
- `Step6_Pearson_r_Comparison.png`
- `Step6_Percent_Overlap_Comparison.png`

#### Step 7 - Integration
- `Step7_CorrelationMatrix_Difference.png`: Correlation matrix difference heatmap

## Statistical Methods

### Tests Used

#### 2-Group Comparison
- **Mann-Whitney U Test**: Non-parametric test for comparing two independent samples
  - Appropriate for non-normally distributed data
  - Null hypothesis: the distributions of both groups are equal
  - Output: U statistic, p-value

#### Multi-Group Comparison (≥3 groups)
- **Kruskal-Wallis Test**: Non-parametric test for comparing ≥3 independent samples
  - Generalizes Mann-Whitney U to multiple groups
  - Null hypothesis: all group distributions are equal
  - Output: H statistic, p-value

#### Post-hoc Pairwise Comparisons
- **Mann-Whitney U Test**: Applied to all pairs
  - **Bonferroni correction**: Adjusted alpha = 0.05 / (number of comparisons)
  - Prevents Type I error inflation from multiple comparisons
  - Reported as: p-value, p-value (Bonferroni corrected)

### Effect Sizes

#### Cohen's d
- Measures standardized difference between group means
- Calculation: d = (mean₁ - mean₂) / pooled_SD
- Interpretation:
  - 0.2 ≤ |d| < 0.5: small effect
  - 0.5 ≤ |d| < 0.8: medium effect
  - |d| ≥ 0.8: large effect

### Significance Levels
- p < 0.001: *** (highly significant)
- p < 0.01: ** (very significant)
- p < 0.05: * (significant)
- p ≥ 0.05: ns (not significant)

## Colorblind-Safe Palette

The tool uses a colorblind-safe palette with 6 colors:
- #0173B2 (blue)
- #DE8F05 (orange)
- #CC79A7 (pink/purple)
- #009E73 (green)
- #D55E00 (red-orange)
- #56B4E9 (light blue)

This palette is optimized for red-blind, green-blind, and blue-yellow blind viewers.

## Examples

### Example 1: Basic Two-Neuron Comparison
```bash
python AMTComparison.py \
  --dirs Composite6 Composite8 \
  --labels "WT" "KO" \
  --output WT_vs_KO_comparison
```

### Example 2: Three-Group Comparison with Analysis
```python
from AMTComparison import ComparativeAnalysis
import pandas as pd

# Setup
ca = ComparativeAnalysis(
    dirs=['WT/', 'KO/', 'Rescue/'],
    labels=['WT', 'KO', 'Rescue']
)

# Run analysis
ca.run_full_analysis('three_group_analysis/')

# Load and inspect results
report = pd.read_csv('three_group_analysis/Comparison_Report.csv')
print(report[report['Metric'] == 'Area'][['Condition', 'Mean', 'Std', 'P_Value']])
```

### Example 3: Custom Visualization with Statistics
```python
from AMTComparison import ComparativeAnalysis

ca = ComparativeAnalysis(dirs=['neuron1/', 'neuron2/'], labels=['Control', 'Treatment'])
ca.set_output_directory('results/')
ca.load_all_data()

# Create specific comparison with statistics
fig, stats = ca.violin_compare(
    metric='Area',
    step_key='Step3',
    title='Mitochondrial Area: Control vs Treatment'
)
fig.savefig('results/area_comparison.png', dpi=600)

# Print statistics
if stats:
    print(f"Test: {stats['test']}")
    print(f"P-value: {stats['p_value']:.4f}")
    print(f"Effect size (Cohen's d): {stats['effect_size']:.3f}")
```

## Citation

If you use AMTComparison in your research, please cite:

```bibtex
@software{AMTComparison2024,
  title={AMTComparison: Advanced Multi-Neuron Analysis Comparison Tool},
  author={AMT Development Team},
  year={2024},
  url={https://github.com/abayatibrain/AMTcomparison}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, questions, or suggestions, please open an issue on the GitHub repository:
https://github.com/abayatibrain/AMTcomparison/issues

## Changelog

### Version 2.0.0 (2024)
- Complete rewrite with class-based architecture
- Multi-neuron comparison support (2+ neurons)
- Statistical testing (Mann-Whitney U, Kruskal-Wallis, Cohen's d)
- Bonferroni multiple comparison correction
- Colorblind-safe visualizations
- Temporal dynamics analysis
- Spatial statistics comparison
- Network topology analysis
- Comprehensive CSV reporting
- Command-line interface
- Python API documentation

### Version 1.0.0 (Original)
- Basic two-neuron comparison
- Step-by-step analysis modules
- Simple visualization functions
