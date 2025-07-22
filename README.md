# From XGBoost to Transformers: 15 Years of Evolving Kaggle Strategies

## Project Overview

This project analyzes the evolution of machine learning tools and techniques over Kaggle's 15-year history using the Meta Kaggle and Meta Kaggle Code datasets. By examining the patterns, trends, and dynamics within this unique ecosystem, we uncover insights into how data science strategies and methodologies have evolved in response to changing technologies, resources, and community knowledge.

## Key Findings

- **Library Adoption**: We trace the rise of gradient boosting libraries (XGBoost, LightGBM, CatBoost) and their eventual competition with deep learning frameworks (PyTorch, TensorFlow) across different competition domains.
  
- **Resource Utilization**: Analysis of GPU usage, execution time, and memory requirements reveals how computational resources correlate with leaderboard performance gains over time.

- **Knowledge Propagation**: Through fork networks and collaboration patterns, we visualize how innovative techniques spread through the Kaggle community.

- **Domain-Specific Evolution**: Different competition domains (Computer Vision, NLP, Tabular) show distinct patterns in tool adoption and methodology trends.

## Interactive Visualizations

Explore our key findings through these interactive visualizations:

1. [Library Adoption Streamgraph](sample/1_streamgraph_library_adoption.html) - Visualizes the rise and evolution of ML libraries from 2010-2025

2. [Compute ROI Heatmap](sample/2_heatmap_compute_roi.html) - Shows the relationship between computation time and performance gains

3. [Methods to Competitions Sankey](sample/3_sankey_methods_competitions.html) - Maps which methods win which types of competitions

4. [Team Diversity Choropleth](sample/4_choropleth_team_diversity.html) - Visualizes team-location diversity versus success metrics

5. [Animated Fork Network](sample/5_animated_fork_network.html) - Shows how knowledge spreads through the community via notebook forks

6. [Library Adoption by Domain](sample/6_library_adoption_by_domain.html) - Compares tool adoption across different competition domains

## Project Structure and Development

### Directory Structure

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

## Technical Methodology

Our analysis pipeline consists of four main components:

1. **Data Layer** - Efficient processing of large-scale Meta Kaggle data using Polars and DuckDB

2. **Feature Extraction** - Code analysis to identify tool usage, techniques, and complexity metrics

3. **Graph Analytics** - NetworkX-based analysis of fork networks and collaboration patterns

4. **Statistical Analysis** - Time series, survival analysis, and regression models to quantify trends

### Setup

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

4. Prepare the dataset directories:

```bash
mkdir -p data/{raw,raw_csv,raw_code,parquet,intermediate,processed,mini_meta}
```

5. Place Meta Kaggle CSV files into `data/raw_csv/` and Meta Kaggle Code files into `data/raw_code/`.

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

The project includes a comprehensive test suite to verify the data pipeline's correctness. Tests are organized into three categories:

1. **Unit Tests**: Test individual functions in isolation
2. **Integration Tests**: Test interactions between pipeline components
3. **End-to-End Tests**: Test the complete pipeline using synthetic data

### Running Tests

You can run all tests with pytest:

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_nodes.py        # Unit tests
pytest tests/test_pipeline_integration.py  # Integration tests
pytest tests/test_pipeline_e2e.py  # End-to-end tests

# Run tests with detailed output
pytest tests/ -v

# Run tests with code coverage report
pytest tests/ --cov=src
```

### Test Structure

- `tests/test_nodes.py`: Tests for individual data processing functions
- `tests/test_pipeline_integration.py`: Tests for pipeline connections
- `tests/test_pipeline_e2e.py`: End-to-end tests with synthetic dataset
- `tests/test_light_pipeline.py`: Light tests that can run without full data

### Writing New Tests

When adding new features to the pipeline, follow these guidelines for test creation:

1. Write unit tests for new functions in `test_nodes.py`
2. Update integration tests if pipeline connections change
3. Ensure end-to-end tests cover the new functionality
4. Use small synthetic datasets to keep tests fast

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
