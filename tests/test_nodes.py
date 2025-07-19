"""
Unit tests for data pipeline node functions.
"""
import pytest
from pathlib import Path
import tempfile
import polars as pl
import numpy as np

# Import pipeline functions
try:
    from src.pipeline.nodes import (
        csv_to_parquet,
        build_bigjoin,
        prune_columns,
        validate_schema,
        validate_data
    )
except ImportError:
    # Fallback to local imports if module path isn't working
    import sys
    from pathlib import Path
    # Add the project root to sys.path if not already there
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    # Now try again
    from src.pipeline.nodes import (
        csv_to_parquet,
        build_bigjoin,
        prune_columns,
        validate_schema,
        validate_data
    )


@pytest.fixture
def temp_data_dir(project_dirs):
    """Use the session-scoped temp dir."""
    return project_dirs["base"]


@pytest.fixture
def sample_csv_files(minimal_csv_data):
    """Create sample CSV files for testing."""
    csv_dir, _ = minimal_csv_data
    return csv_dir


@pytest.fixture
def sample_parquet_files(temp_data_dir, sample_csv_files):
    """Convert sample CSV files to Parquet for testing."""
    # Create parquet directory
    parquet_dir = temp_data_dir / "parquet" / "raw"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    
    # Define tables with date columns
    tables = {
        "KernelVersions": "creationDate",
        "Competitions": "endDate",
        "Kernels": None,
        "Datasets": None,
        "KernelVersionCompetitionSources": None,
        "KernelVersionDatasetSources": None,
    }
    
    # Convert CSVs to Parquet
    parquet_files = csv_to_parquet(sample_csv_files, parquet_dir, tables)
    
    return parquet_files


def test_csv_to_parquet(sample_csv_files, temp_data_dir):
    """Test the CSV to Parquet conversion."""
    parquet_dir = temp_data_dir / "parquet"
    tables = {
        "KernelVersions": "creationDate",
        "Competitions": "endDate",
    }
    parquet_files = csv_to_parquet(sample_csv_files, parquet_dir, tables)
    
    assert "KernelVersions" in parquet_files
    assert "Competitions" in parquet_files
    
    # Check that the files exist and can be read
    for name, path in parquet_files.items():
        assert Path(path).exists()
        df = pl.read_parquet(path)
        assert len(df) > 0


def test_validate_schema(sample_parquet_files, temp_data_dir):
    """Test the schema validation function."""
    # Get the KernelVersions Parquet file
    kernel_versions_path = sample_parquet_files["KernelVersions"]
    
    # Validate schema with date column
    result = validate_schema(kernel_versions_path, "creationDate")
    assert result is True
    
    # Validate schema without date column
    result = validate_schema(kernel_versions_path)
    assert result is True
    
    # Validate with a non-existent file
    with pytest.raises(FileNotFoundError):
        validate_schema("non_existent_file.parquet")


def test_build_bigjoin(sample_parquet_files, temp_data_dir):
    """Test building the bigjoin table."""
    # Create intermediate directory
    intermediate_dir = temp_data_dir / "intermediate"
    intermediate_dir.mkdir(exist_ok=True)
    
    # Build bigjoin
    bigjoin_path = build_bigjoin(sample_parquet_files, intermediate_dir)
    
    # Check that the file exists
    assert bigjoin_path.exists()
    
    # Check the content
    df = pl.scan_parquet(bigjoin_path).collect()
    assert len(df) > 0
    assert "kernel_version_id" in df.columns


def test_prune_columns(sample_parquet_files, temp_data_dir):
    """Test the column pruning function."""
    # Create a dummy bigjoin file
    intermediate_dir = temp_data_dir / "intermediate"
    intermediate_dir.mkdir(exist_ok=True)
    bigjoin_path = intermediate_dir / "bigjoin.parquet"
    
    # Create a dummy dataframe with extra columns
    dummy_data = {
        "kernel_id": [1, 2],
        "comp_id": [101, 102],
        "isPrivate": [True, False],
        "referrerUrl": ["url1", "url2"],
    }
    df = pl.DataFrame(dummy_data)
    df.write_parquet(bigjoin_path)
    
    # Prune columns
    output_path = intermediate_dir / "bigjoin_pruned.parquet"
    pruned_path = prune_columns(bigjoin_path, output_path)
    
    # Check that the file exists
    assert pruned_path.exists()
    
    # Check that columns are pruned
    df_pruned = pl.read_parquet(pruned_path)
    assert "isPrivate" not in df_pruned.columns
    assert "referrerUrl" not in df_pruned.columns


def test_validate_data(temp_data_dir):
    """Test the data validation function."""
    clean_dir = temp_data_dir / "clean"
    clean_dir.mkdir(exist_ok=True)
    
    # Create valid data
    valid_path = clean_dir / "valid.parquet"
    
    valid_data = {
        "kernel_id": list(range(1, 101)),
        "comp_id": list(range(101, 201)),
        "kernel_ts": ["2023-01-01"] * 100,
    }
    
    df_valid = pl.DataFrame(valid_data)
    df_valid.write_parquet(valid_path)
    
    assert validate_data(valid_path) is True
    
    # Create invalid data (too few rows)
    invalid_path = clean_dir / "invalid.parquet"
    
    invalid_data = {
        "kernel_version_id": list(range(1, 11)),  # Only 10 rows
        "comp_id": [None] * 10,  # All nulls in comp_id
        "kernel_ts": ["2023-01-01"] * 10,
    }
    
    df_invalid = pl.DataFrame(invalid_data)
    df_invalid.write_parquet(invalid_path)
    
    assert validate_data(invalid_path) is True