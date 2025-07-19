"""
Integration tests for the data pipeline.
"""
import pytest
import os
from pathlib import Path
import polars as pl
import shutil
from kedro.framework.context import KedroContext
from kedro.framework.startup import bootstrap_project
from kedro.framework.session import KedroSession
from kedro.io import DataCatalog
from kedro.pipeline import Pipeline, node
from kedro.framework.project import settings
from kedro.config import OmegaConfigLoader
import time
from kedro.io import MemoryDataset
from kedro.runner import SequentialRunner

# Import Kedro wrapper node functions that accept the `params:etl` input.
try:
    from src.pipeline.etl import (
        csv_to_parquet_node as csv_to_parquet_node,
        build_bigjoin_node as build_bigjoin_node,
    )
except ImportError:
    pytestmark = pytest.mark.skip(reason="ETL wrapper node functions not found")


@pytest.fixture
def catalog(project_dirs):
    """Create a catalog for testing."""
    return DataCatalog({
        "params:etl": MemoryDataset(
            {
                "src_dir": str(project_dirs["raw_csv"]),
                "dst_dir": str(project_dirs["parquet_raw"]),
                "intermediate_dir": str(project_dirs["intermediate"]),
                "mini_meta_dir": str(project_dirs["mini_meta"]),
                "sample_frac": 0.01,
                "tables": {
                    "KernelVersions": "creationDate",
                    "Competitions": "endDate",
                    "Kernels": None,
                    "KernelVersionCompetitionSources": None,
                    "KernelVersionDatasetSources": None,
                    "Datasets": None,
                },
            }
        ),
        "parquet_raw": MemoryDataset(),
        "bigjoin": MemoryDataset(),
    })


@pytest.fixture(autouse=True)
def create_test_data(project_dirs):
    """Create sample CSV files for testing."""
    csv_dir = project_dirs["raw_csv"]
    
    # Create a sample KernelVersions.csv
    kernel_versions_content = """Id,AuthorUserId,CurrentKernelVersionId,ForkParentKernelVersionId,ForumTopicId,FirstKernelVersionId,TotalViews,TotalComments,TotalVotes,isCommit,creationDate
1,100,1,NULL,NULL,1,150,10,25,True,2023-01-01
2,101,2,NULL,NULL,2,200,15,30,True,2023-01-02
3,102,3,1,NULL,3,250,20,35,False,2023-01-03"""
    
    with open(csv_dir / "KernelVersions.csv", "w") as f:
        f.write(kernel_versions_content)
    
    # Create a sample Competitions.csv
    competitions_content = """Id,Title,EnabledDate,endDate
101,Test Competition 1,2023-01-01,2023-06-30
102,Test Competition 2,2023-01-01,2023-07-31"""
    
    with open(csv_dir / "Competitions.csv", "w") as f:
        f.write(competitions_content)
    
    # Create source relationship tables
    kvcs_content = """KernelVersionId,SourceCompetitionId
1,101
2,102"""
    
    with open(csv_dir / "KernelVersionCompetitionSources.csv", "w") as f:
        f.write(kvcs_content)
    
    kvds_content = """KernelVersionId,SourceDatasetVersionId,datasetId
1,201,301
2,202,302"""
    
    with open(csv_dir / "KernelVersionDatasetSources.csv", "w") as f:
        f.write(kvds_content)

    # Create a sample Kernels.csv
    kernels_content = """Id,AuthorUserId,CurrentKernelVersionId,ForkParentKernelVersionId,ForumTopicId,FirstKernelVersionId,TotalViews,TotalComments,TotalVotes
1001,100,1,NULL,NULL,1,150,10,25
1002,101,2,NULL,NULL,2,200,15,30
1003,102,3,1,NULL,3,250,20,35"""

    with open(csv_dir / "Kernels.csv", "w") as f:
        f.write(kernels_content)

    # Create a sample Datasets.csv
    datasets_content = """Id,CreatorUserId,OwnerUserId,Title
201,100,100,Test Dataset 1
202,101,101,Test Dataset 2"""

    with open(csv_dir / "Datasets.csv", "w") as f:
        f.write(datasets_content)


@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline for testing."""
    return Pipeline(
        [
            node(
                csv_to_parquet_node,  # Wrapper that expects params only
                inputs="params:etl",
                outputs="parquet_raw",
                name="csv_to_parquet_node",
            ),
            node(
                build_bigjoin_node,  # Wrapper that expects parquet_raw and params
                inputs=["parquet_raw", "params:etl"],
                outputs="bigjoin",
                name="build_bigjoin_node",
            ),
        ]
    )


def test_csv_to_parquet_node_integration(catalog, mock_pipeline):
    """Test that the csv_to_parquet node works and produces valid output files."""
    # Run just the first node
    pipeline = Pipeline([mock_pipeline.nodes[0]])
    runner = SequentialRunner()
    runner.run(pipeline, catalog)

    # Verify the node output contains expected keys
    loaded_output = catalog.load("parquet_raw")
    assert "KernelVersions" in loaded_output
    assert "Competitions" in loaded_output


def test_build_bigjoin_node_integration(catalog, mock_pipeline):
    """Test that the build_bigjoin node works correctly with the csv_to_parquet node output."""
    # Run the first two nodes
    pipeline = Pipeline(mock_pipeline.nodes[:2])
    runner = SequentialRunner()
    runner.run(pipeline, catalog)
    
    # Verify the file was created
    bigjoin_path = catalog.load("bigjoin")
    assert bigjoin_path.exists()
    
    # Check that we can read it and it has expected columns
    df = pl.scan_parquet(bigjoin_path).collect()
    assert "kernel_version_id" in df.columns
    assert "comp_id" in df.columns
    assert "dataset_id" in df.columns