# How Hard is Microservice Anomaly Detection?

## Overview

This repository contains the complete replication package for an empirical study evaluating anomaly detection methods on microservice trace benchmarks. Key findings:

1. **Benchmark may be too easy**: Random Forest achieves 99.95% F1 using only 5 aggregated features
2. **Fragile generalization**: All models degrade to 43% F1 under distribution shift
3. **Simple solution works**: Feature normalization recovers 99% F1 for cross-system deployment

## Repository Structure

```
xhybrid-2026/
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── features/          # Feature extraction (SAX, aggregation)
│   ├── models/            # Model implementations
│   ├── training/          # Training pipelines
│   └── xai/               # Explainability (SHAP, attention)
├── scripts/               # Experiment scripts
│   ├── run_cv_experiments.py      # 5-fold cross-validation
│   ├── cross_dataset_validation.py # Distribution shift experiments
│   ├── xai_shap_analysis.py       # SHAP feature importance
│   ├── ablation_real_xhybrid.py   # Ablation study
│   └── regenerate_all_figures.py  # Figure generation
├── data/                  # Dataset (download instructions below)
├── figures/               # Generated figures
└── requirements.txt       # Python dependencies
```

## Dataset

This study uses the **RE2-TT dataset** from the RCAEval benchmark:

- **Source**: [Zenodo](https://doi.org/10.5281/zenodo.14590730)
- **Size**: 573,945 traces, 67.3M spans
- **Services**: 27 microservices
- **Fault Types**: 6 (CPU, Memory, Disk, Network, Packet Loss, Socket)

### Download Instructions

```bash
# Download from Zenodo
wget https://zenodo.org/records/14590730/files/train-ticket.zip
unzip train-ticket.zip -d data/raw/
```

## Installation

```bash
# Clone repository
cd xhybrid-2026

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Reproducing Results

### 1. Main Experiments (Table 3 & 4)

```bash
# Run 5-fold cross-validation for all models
python scripts/run_cv_experiments.py

# Output: results/cv_results.json
```

### 2. Cross-Dataset Validation (Table 7)

```bash
# Run distribution shift experiments
python scripts/cross_dataset_validation.py

# Output: results/cross_dataset_results.json
```

### 3. SHAP Analysis (Figure 5)

```bash
# Generate SHAP feature importance
python scripts/xai_shap_analysis.py

# Output: figures/shap_simple.pdf
```

### 4. Ablation Study (Table 5)

```bash
# Run ablation experiments
python scripts/ablation_real_xhybrid.py

# Output: results/ablation_results.json
```

### 5. Regenerate All Figures

```bash
# Generate all paper figures
python scripts/regenerate_all_figures.py

# Output: figures/*.pdf
```

## Models Evaluated

| Model | Description | F1 (%) |
|-------|-------------|--------|
| Random Forest | 100 trees, max depth 10 | 99.95 |
| XGBoost | 100 estimators, max depth 6 | 99.93 |
| Multi-Branch | 3-branch neural architecture | 99.43 |
| Logistic Regression | L2 regularization | 96.41 |
| Isolation Forest | Unsupervised, contamination=0.1 | 29.10 |
| One-Class SVM | RBF kernel | 34.20 |

## Features Used

The study extracts 5 aggregated trace-level features:

1. `num_spans` - Number of spans in trace
2. `avg_duration` - Average span duration (ms)
3. `total_duration` - Total trace duration (ms)
4. `max_duration` - Maximum span duration (ms)
5. `num_services` - Number of unique services

## Hardware

Experiments conducted on:
- CPU: AMD Ryzen 9 5900X (12 cores)
- RAM: 64GB
- GPU: NVIDIA RTX 3080
- OS: Ubuntu 22.04 / macOS 14

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.