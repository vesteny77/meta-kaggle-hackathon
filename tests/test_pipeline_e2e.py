"""
End-to-end tests for the data pipeline using a small synthetic dataset.
"""
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import polars as pl
import pytest
import time
from contextlib import contextmanager

# Import pipeline functions
try:
    from src.pipeline.nodes import (
        build_bigjoin,
        create_mini_meta,
        csv_to_parquet,
        prune_columns,
        validate_data,
        validate_schema,
    )
except ImportError:
    # Mark all tests in this module to be skipped if imports fail
    pytestmark = pytest.mark.skip(reason="Pipeline nodes module not found")


@pytest.fixture
def e2e_test_dir(project_dirs):
    """Create a temporary directory for E2E testing."""
    return project_dirs["base"]


@contextmanager
def timing(description: str) -> None:
    """A context manager to time a block of code."""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"{description} took {elapsed:.2f} seconds")


def test_e2e_pipeline(e2e_test_dir):
    """Run the entire pipeline end-to-end with synthetic data."""
    data_dir = e2e_test_dir / "data"
    csv_dir = data_dir / "raw_csv"
    parquet_dir = data_dir / "parquet/raw"
    intermediate_dir = data_dir / "intermediate"
    mini_meta_dir = data_dir / "mini_meta"

    # Define tables with date columns
    tables = {
        "KernelVersions": "creationDate",
        "Competitions": "endDate",
        "Datasets": "creationDate",
        "Users": "creationDate",
        "Kernels": None,
        "KernelVersionCompetitionSources": None,
        "KernelVersionDatasetSources": None,
    }

    # 1. Test CSV to Parquet conversion
    with timing("CSV to Parquet conversion"):
        parquet_files = csv_to_parquet(csv_dir, parquet_dir, tables)

    # Verify output
    for table in tables:
        assert (parquet_dir / f"{table}.parquet").exists(), f"Missing Parquet file for {table}"

    # 2. Validate schema
    with timing("Schema validation"):
        schema_valid = validate_schema(parquet_files["KernelVersions"], "creationDate")

    assert schema_valid is True, "Schema validation failed"

    # 3. Build bigjoin
    with timing("Bigjoin creation"):
        bigjoin_path = build_bigjoin(parquet_files, intermediate_dir)

    assert bigjoin_path.exists(), "Bigjoin file not created"

    # Check bigjoin content
    bigjoin_df = pl.scan_parquet(bigjoin_path).collect()
    assert len(bigjoin_df) > 0, "Bigjoin is empty"
    assert "kernel_version_id" in bigjoin_df.columns
    assert "comp_id" in bigjoin_df.columns

    # 4. Prune columns
    with timing("Column pruning"):
        clean_path = intermediate_dir / "kernel_bigjoin_clean.parquet"
        pruned_path = prune_columns(bigjoin_path, clean_path)

    assert pruned_path.exists(), "Pruned file not created"

    # Verify column pruning
    pruned_df = pl.scan_parquet(pruned_path).collect()
    assert "referrerUrl" not in pruned_df.columns
    assert "isPrivate" not in pruned_df.columns

    # 5. Validate cleaned data
    with timing("Data validation"):
        data_valid = validate_data(pruned_path)

    assert data_valid is True, "Data validation failed"

    # 6. Create mini-meta
    with timing("Mini-meta creation"):
        mini_meta_path = create_mini_meta(
            pruned_path, mini_meta_dir / "mini_meta.parquet"
        )

    assert mini_meta_path.exists(), "Mini-meta file not created"


def test_e2e_data_consistency(e2e_test_dir):
    """Test data consistency across the pipeline stages."""
    # Run the pipeline first
    data_dir = e2e_test_dir / "data"
    csv_dir = data_dir / "raw_csv"
    parquet_dir = data_dir / "parquet/raw"
    intermediate_dir = data_dir / "intermediate"
    mini_meta_dir = data_dir / "mini_meta"

    # Define tables with date columns
    tables = {
        "KernelVersions": "creationDate",
        "Competitions": "endDate",
        "Datasets": "creationDate",
        "Users": "creationDate",
        "Kernels": None,
        "KernelVersionCompetitionSources": None,
        "KernelVersionDatasetSources": None,
    }

    # Run pipeline
    parquet_files = csv_to_parquet(csv_dir, parquet_dir, tables)
    time.sleep(1)
    bigjoin_path = build_bigjoin(parquet_files, intermediate_dir)
    clean_path = intermediate_dir / "kernel_bigjoin_clean.parquet"
    pruned_path = prune_columns(bigjoin_path, clean_path)

    # Now test consistency

    # Compare CSV to Parquet counts
    kv_csv = pd.read_csv(csv_dir / "KernelVersions.csv")
    kv_parquet = pl.read_parquet(parquet_files["KernelVersions"])

    assert len(kv_csv) == len(kv_parquet), "Row count mismatch between CSV and Parquet"

    # Check committed kernels match in raw data and bigjoin
    committed_csv = kv_csv[kv_csv.isCommit == True].shape[0]
    committed_bigjoin = pl.scan_parquet(bigjoin_path).filter(pl.col("is_commit") == True).collect().height
    assert committed_csv == committed_bigjoin, "Committed kernel count mismatch"

    # Check that pruned columns are gone
    pruned_df = pl.read_parquet(pruned_path)
    assert "isPrivate" not in pruned_df.columns
    
    # Check primary keys are preserved
    kernel_ids_parquet = set(kv_parquet["Id"].to_list())
    kernel_ids_bigjoin = set(pl.read_parquet(bigjoin_path)["kernel_version_id"].to_list())
    assert kernel_ids_parquet == kernel_ids_bigjoin, "Primary key mismatch"


def test_pipeline_with_bad_data(e2e_test_dir):
    """Test pipeline robustness with intentionally corrupted data."""
    data_dir = e2e_test_dir / "data"
    csv_dir = data_dir / "raw_csv"
    bad_csv_dir = data_dir / "bad_csv"
    parquet_dir = data_dir / "parquet/raw"
    intermediate_dir = data_dir / "intermediate"

    # Copy existing data to bad_csv_dir
    bad_csv_dir.mkdir(exist_ok=True)
    for csv_file in csv_dir.glob("*.csv"):
        shutil.copy(csv_file, bad_csv_dir / csv_file.name)

    # Corrupt the KernelVersions.csv file by removing creationDate column
    kernel_versions_path = bad_csv_dir / "KernelVersions.csv"
    df = pd.read_csv(kernel_versions_path)
    df = df.drop("creationDate", axis=1)
    df.to_csv(kernel_versions_path, index=False)

    # Define tables
    tables = {
        "KernelVersions": "creationDate",  # This column no longer exists
        "Competitions": "endDate",
        "Kernels": None,
        "KernelVersionCompetitionSources": None,
        "KernelVersionDatasetSources": None,
    }

    # Attempt to run the pipeline with bad data
    try:
        parquet_files = csv_to_parquet(bad_csv_dir, parquet_dir, tables)
        # Even with missing date column, conversion should not happen for KernelVersions
        assert "KernelVersions" not in parquet_files

    except Exception as e:
        # If csv_to_parquet fails entirely (which is also acceptable), verify it was due to missing column
        pytest.fail(f"csv_to_parquet failed with an unexpected error: {str(e)}")

    # Test with empty relationship files
    empty_bad_dir = data_dir / "empty_bad_csv"
    empty_bad_dir.mkdir(exist_ok=True)
    for csv_file in csv_dir.glob("*.csv"):
        shutil.copy(csv_file, empty_bad_dir / csv_file.name)
    
    # Empty the relationship files
    (empty_bad_dir / "KernelVersionCompetitionSources.csv").write_text("KernelVersionId,SourceCompetitionId\n")
    (empty_bad_dir / "KernelVersionDatasetSources.csv").write_text("KernelVersionId,SourceDatasetVersionId\n")

    # Try pipeline with missing relationship data
    try:
        parquet_files = csv_to_parquet(empty_bad_dir, parquet_dir, tables)
        time.sleep(1)
        # Should succeed but produce empty relationship files
        bigjoin_path = build_bigjoin(parquet_files, intermediate_dir)

        # Bigjoin should have kernels but fewer competition relationships
        bigjoin_df = pl.scan_parquet(bigjoin_path).collect()
        assert len(bigjoin_df) > 0
        assert bigjoin_df["comp_id"].is_null().all()

    except Exception as e:
        pytest.fail(f"Pipeline failed with empty relationship files: {str(e)}")