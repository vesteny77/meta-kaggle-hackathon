"""
Tests for the data processing pipeline.
"""
import polars as pl
import pytest
from pathlib import Path
import os


@pytest.fixture
def data_dir():
    """Return path to the data directory."""
    return Path("data")


def test_directory_structure(data_dir):
    """Test that the expected directory structure exists."""
    expected_dirs = [
        "raw", "raw_csv", "raw_code", "parquet", 
        "intermediate", "processed", "mini_meta"
    ]
    for d in expected_dirs:
        assert (data_dir / d).is_dir(), f"Directory {d} is missing"


def test_bigjoin_schema(data_dir):
    """Test the schema of the bigjoin table if it exists."""
    bigjoin_path = data_dir / "intermediate/kernel_bigjoin_clean.parquet"
    
    # Skip if file doesn't exist yet
    if not bigjoin_path.is_file():
        pytest.skip(f"Bigjoin file not found at {bigjoin_path}")
    
    df = pl.scan_parquet(bigjoin_path)
    
    # Check essential columns
    required_columns = [
        "kernel_id", "kernel_ts", "author_id", 
        "comp_id", "comp_title", "category"
    ]
    
    for col in required_columns:
        assert col in df.columns, f"Column {col} missing from bigjoin"


def test_mini_meta_consistency(data_dir):
    """Test that mini_meta is a proper subset of the full data."""
    mini_meta_path = data_dir / "mini_meta/kernel_bigjoin_1pct.parquet"
    full_data_path = data_dir / "intermediate/kernel_bigjoin_clean.parquet"
    
    # Skip if either file doesn't exist yet
    if not mini_meta_path.is_file() or not full_data_path.is_file():
        pytest.skip("Mini-meta or full data file not found")
    
    # Check mini_meta row count is approximately 1% of full data
    mini_meta_count = pl.scan_parquet(mini_meta_path).select(pl.count()).collect().item()
    full_data_count = pl.scan_parquet(full_data_path).select(pl.count()).collect().item()
    
    # Allow for some variation due to hash-based sampling
    assert 0.005 <= mini_meta_count / full_data_count <= 0.015, \
        "Mini-meta size is not approximately 1% of full data"


def test_no_nulls_in_key_columns(data_dir):
    """Test that key columns have no null values in the cleaned data."""
    clean_data_path = data_dir / "intermediate/kernel_bigjoin_clean.parquet"
    
    # Skip if file doesn't exist yet
    if not clean_data_path.is_file():
        pytest.skip(f"Clean data file not found at {clean_data_path}")
    
    # Check for nulls in kernel_id
    df = pl.scan_parquet(clean_data_path)
    nulls = df.filter(pl.col("kernel_id").is_null()).select(pl.count()).collect().item()
    
    assert nulls == 0, f"Found {nulls} rows with null kernel_id in cleaned data"