# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2024-03-12

### Added
- **Multi-neuron comparison**: Support for comparing 2 or more neurons (was limited to 2)
- **Statistical testing suite**:
  - Mann-Whitney U test for pairwise comparisons
  - Kruskal-Wallis H test for multi-group comparisons (≥3 groups)
  - Cohen's d effect size calculation
  - Bonferroni multiple comparison correction for pairwise tests
- **New comparison modules**:
  - `compare_temporal_dynamics()`: Analyze temporal stability and trends
  - `compare_spatial_statistics()`: Compare spatial distributions and clustering
  - `compare_network_topology()`: Analyze network connectivity metrics
- **Enhanced reporting**:
  - `generate_comparison_report()`: Comprehensive CSV report with all metrics and statistics
  - Temporal dynamics summary CSV
  - Spatial statistics summary CSV
  - Network topology summary CSV
- **Improved visualizations**:
  - Colorblind-safe palette (6 colors optimized for all color vision deficiencies)
  - Significance annotations on plots (* p<0.05, ** p<0.01, *** p<0.001)
  - Effect size display (Cohen's d values)
  - 600 DPI output for publication-quality figures
  - Automatic scaling based on number of neurons
- **Class-based architecture**: `ComparativeAnalysis` class for better code organization and reusability
- **Command-line interface**: Full argparse CLI with short and long options
  - `--dirs` / `-d`: Input directories
  - `--labels` / `-l`: Custom labels for neurons
  - `--output` / `-o`: Output directory
- **Python API**: Clean, documented API for programmatic access
  - `ComparativeAnalysis.__init__()`: Initialize with multiple neurons
  - `set_output_directory()`: Configure output path
  - `load_all_data()`: Load all available data
  - `run_full_analysis()`: Execute complete analysis pipeline
  - `violin_compare()`: Create comparison plots with statistics
- **Documentation**:
  - Comprehensive README.md with installation, usage, and API documentation
  - EXAMPLES.md with detailed use cases and advanced examples
  - Inline docstrings for all public methods
  - Module-level documentation
- **Package structure**:
  - `setup.py` for package installation and distribution
  - Console script entry point (`amt-compare`)
  - Proper package metadata

### Changed
- **Rewritten core functionality**: From procedural script to object-oriented design
- **Refactored utility functions**:
  - Enhanced `load_csv_safe()` with better error handling
  - New `cohen_d()` for effect size calculation
  - New `get_significance_stars()` for p-value annotation
  - New `add_stat_annotation()` for plot annotation
- **Improved default color palette**: Changed from basic blue/orange to colorblind-safe palette
- **Enhanced error handling**: More informative error messages and graceful degradation
- **Better data validation**: Automatic NaN handling and numeric coercion
- **Improved plot quality**: 600 DPI by default (was 300), better font sizing

### Improved
- **Code organization**: Separated concerns into logical methods
- **Performance**: Optimized data loading and processing
- **User experience**: Better progress feedback and informative output
- **Accessibility**: Colorblind-safe visualizations by default
- **Robustness**: Better handling of missing files and incomplete data

### Removed
- IPython/Jupyter dependencies (Jupyter-only display functions)
- Hard-coded directory names (DIR1, DIR2, LABEL1, LABEL2 variables)
- Global configuration variables (now passed to ComparativeAnalysis class)
- Direct matplotlib show() calls (all output to files)

## [1.0.0] - 2023

### Initial Release
- Basic two-neuron comparison functionality
- Step-by-step analysis for Steps 1-7
- Violin plots for metric comparison
- Simple heatmap for correlation matrices
- Utility functions for CSV loading
- Jupyter notebook-based workflow
