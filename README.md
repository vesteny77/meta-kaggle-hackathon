# Meta Kaggle Hackathon Project Setup

This document describes how to set up and run the Meta Kaggle Hackathon project.

## Project Overview

From XGBoost to Transformers: 15 Years of Evolving Kaggle Strategies

This project analyzes the evolution of machine learning tools and techniques over Kaggle's history using the Meta Kaggle and Meta Kaggle Code datasets.

## Directory Structure

```
meta-kaggle-hackathon/
├── conf/                   # Kedro configuration
│   ├── base/               # Base configuration
│   └── local/              # Local configuration (git-ignored)
├── data/                   # All data files
│   ├── raw/                # Raw data
│   ├── raw_csv/            # Meta Kaggle CSV files
│   ├── raw_code/           # Meta Kaggle Code files
│   ├── parquet/            # Converted Parquet files
│   ├── intermediate/       # Intermediate processing files
│   ├── processed/          # Final processed data
│   └── mini_meta/          # 1% samples for testing
├── notebooks/              # Jupyter notebooks
├── scripts/                # Utility scripts
├── src/                    # Source code
│   ├── pipeline/           # Kedro pipeline definitions
│   ├── features/           # Feature extraction
│   ├── graphs/             # NetworkX graph analysis
│   ├── stats/              # Statistical analysis
│   └── app/                # Dash visualization app
├── tests/                  # Unit tests
│   └── fixtures/           # Test fixtures
└── visuals/                # Generated visualizations
    ├── static/             # Static images for reports
    ├── html/               # Interactive HTML visualizations
    ├── dash_assets/        # Dash app assets
    └── video_frames/       # Video frames for animations
```

## Setup

1. Make sure you have Conda installed (Miniconda or Anaconda)

2. Set up the environment:

```bash
# Make the script executable
chmod +x setup_env.sh

# Run the setup script
./setup_env.sh
```

3. Activate the environment:

```bash
conda activate meta-kaggle
```

3. Prepare the dataset directories:

```bash
mkdir -p data/{raw,raw_csv,raw_code,parquet,intermediate,processed,mini_meta}
```

4. Place Meta Kaggle CSV files into `data/raw_csv/` and Meta Kaggle Code files into `data/raw_code/`.

## Running the Pipeline

1. Convert CSV to Parquet:

```bash
python scripts/csv_to_parquet.py
```

2. Build the bigjoin:

```bash
python scripts/build_bigjoin.py
```

3. Clean and prune columns:

```bash
python scripts/prune_columns.py
```

4. Create mini-meta sample:

```bash
python scripts/make_mini_meta.py --frac 0.01
```

5. Run the full pipeline:

```bash
python scripts/run_data_pipeline.py
```

6. Or run the Kedro pipeline:

```bash
python -m src.main etl
```

## Testing

Run tests with pytest:

```bash
pytest tests/
```

## Development

1. Set up pre-commit hooks:

```bash
pre-commit install
```

2. Format and lint code before committing:

```bash
black src/
ruff check --fix src/
mypy src/
```