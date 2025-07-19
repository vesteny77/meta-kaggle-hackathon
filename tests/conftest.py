"""
Pytest configuration file for meta-kaggle-hackathon project.

This file contains shared fixtures and configuration for the test suite.
"""
import pytest
from pathlib import Path
import tempfile
import polars as pl


@pytest.fixture(scope="session")
def session_temp_dir(tmpdir_factory):
    """
    Create a session-scoped temporary directory for test artifacts.
    
    This ensures that all tests in a session share the same base temp directory,
    which is useful for integration tests where one test creates files that
    another test needs to read.
    
    Returns:
        Path: Path to the session-scoped temporary directory.
    """
    return Path(tmpdir_factory.mktemp("meta_kaggle_test_session"))


@pytest.fixture
def project_dirs(session_temp_dir):
    """
    Create the necessary subdirectory structure for a test project.
    
    Args:
        session_temp_dir: The session-scoped temporary directory fixture.
    
    Returns:
        dict: A dictionary of paths to the created directories.
    """
    dirs = {
        "base": session_temp_dir,
        "raw_csv": session_temp_dir / "data" / "raw_csv",
        "parquet_raw": session_temp_dir / "data" / "parquet" / "raw",
        "intermediate": session_temp_dir / "data" / "intermediate",
        "mini_meta": session_temp_dir / "data" / "mini_meta",
        "conf_base": session_temp_dir / "conf" / "base",
    }
    
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
        
    return dirs


@pytest.fixture
def minimal_csv_data(project_dirs):
    """
    Create minimal CSV files required for testing.
    
    Args:
        project_dirs: The project directory structure fixture.
    
    Returns:
        tuple: (csv_dir, expected_row_counts) with csv directory path and row counts
    """
    csv_dir = project_dirs["raw_csv"]
    
    # Create KernelVersions.csv
    kernel_versions_csv = """Id,creationDate,authorUserId,totalTokens,execSeconds,gpuType,isCommit,parentId
1,2023-01-01,100,1000,30,Tesla P100,True,
2,2023-01-02,101,1200,45,None,True,
3,2023-01-03,102,1500,60,None,False,1"""
    with open(csv_dir / "KernelVersions.csv", "w") as f:
        f.write(kernel_versions_csv)

    # Create Competitions.csv
    competitions_csv = """Id,Title,category,endDate
101,Test Competition 1,CV,2023-06-30
102,Test Competition 2,NLP,2023-07-31"""
    with open(csv_dir / "Competitions.csv", "w") as f:
        f.write(competitions_csv)

    # Create KernelVersionCompetitionSources.csv
    kvcs_csv = """Id,KernelVersionId,SourceCompetitionId
1,1,101
2,2,102"""
    with open(csv_dir / "KernelVersionCompetitionSources.csv", "w") as f:
        f.write(kvcs_csv)

    # Create KernelVersionDatasetSources.csv
    kvds_csv = """Id,KernelVersionId,SourceDatasetVersionId
1,1,201
2,2,202"""
    with open(csv_dir / "KernelVersionDatasetSources.csv", "w") as f:
        f.write(kvds_csv)

    # Create Kernels.csv
    kernels_csv = """Id,AuthorUserId,CurrentKernelVersionId,ForkParentKernelVersionId,ForumTopicId,FirstKernelVersionId,TotalViews,TotalComments,TotalVotes
1001,100,1,NULL,NULL,1,150,10,25
1002,101,2,NULL,NULL,2,200,15,30
1003,102,3,1,NULL,3,250,20,35"""
    with open(csv_dir / "Kernels.csv", "w") as f:
        f.write(kernels_csv)

    # Create Datasets.csv
    datasets_csv = """Id,CreatorUserId,OwnerUserId,Title
201,100,100,Test Dataset 1
202,101,101,Test Dataset 2"""
    with open(csv_dir / "Datasets.csv", "w") as f:
        f.write(datasets_csv)

    # Create Users.csv
    users_csv = """Id,UserName,DisplayName,RegisterDate
100,user1,User One,2023-01-01
101,user2,User Two,2023-01-01
102,user3,User Three,2023-01-01"""
    with open(csv_dir / "Users.csv", "w") as f:
        f.write(users_csv)
    
    # Expected row counts
    row_counts = {
        "KernelVersions": 3,
        "Competitions": 2,
        "KernelVersionCompetitionSources": 2,
        "KernelVersionDatasetSources": 2,
        "Kernels": 3,
        "Datasets": 2,
        "Users": 3,
    }
    
    return csv_dir, row_counts