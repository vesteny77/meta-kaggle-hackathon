"""
ETL pipeline for the Meta Kaggle project.
"""
from kedro.pipeline import Pipeline, node
from pathlib import Path
from .nodes import csv_to_parquet, build_bigjoin, prune_columns, create_mini_meta


def csv_to_parquet_node(params):
    """
    Node function wrapper for csv_to_parquet.
    
    Args:
        params: Pipeline parameters
        
    Returns:
        dict: Dictionary of output Parquet file paths
    """
    src_dir = Path(params.get("src_dir", "data/raw_csv"))
    dst_dir = Path(params.get("dst_dir", "data/parquet/raw"))
    tables = params.get("tables", {
        "KernelVersions": "CreationDate",
        "Competitions": "DeadlineDate",
        "Datasets": "CreationDate",
        "KernelVersionCompetitionSources": "",
        "KernelVersionDatasetSources": "",
        "Users": "RegisterDate",
        "ForumMessages": "PostDate"
    })
    
    return csv_to_parquet(src_dir, dst_dir, tables)


def build_bigjoin_node(parquet_raw, params):
    """
    Node function wrapper for build_bigjoin.
    
    Args:
        parquet_raw: Dictionary of input Parquet file paths
        params: Pipeline parameters
        
    Returns:
        Path: Path to the output bigjoin Parquet file
    """
    output_dir = Path(params.get("intermediate_dir", "data/intermediate"))
    return build_bigjoin(parquet_raw, output_dir)


def prune_columns_node(bigjoin, params):
    """
    Node function wrapper for prune_columns.
    
    Args:
        bigjoin: Path to the input bigjoin Parquet file
        params: Pipeline parameters
        
    Returns:
        Path: Path to the output cleaned Parquet file
    """
    intermediate_dir = Path(params.get("intermediate_dir", "data/intermediate"))
    output_path = intermediate_dir / "kernel_bigjoin_clean.parquet"
    return prune_columns(Path(bigjoin), output_path)


def create_mini_meta_node(bigjoin_clean, params):
    """
    Node function wrapper for create_mini_meta.
    
    Args:
        bigjoin_clean: Path to the cleaned bigjoin file
        params: Pipeline parameters
        
    Returns:
        Path: Path to the mini-meta sample
    """
    sample_frac = params.get("sample_frac", 0.01)
    mini_meta_dir = Path(params.get("mini_meta_dir", "data/mini_meta"))
    output_path = mini_meta_dir / f"kernel_bigjoin_{int(sample_frac*100)}pct.parquet"
    return create_mini_meta(Path(bigjoin_clean), output_path, sample_frac)


def create_pipeline(**kwargs):
    """Create the ETL pipeline."""
    return Pipeline(
        [
            node(
                csv_to_parquet_node,
                inputs="params:etl",
                outputs="parquet_raw",
                name="csv_to_parquet",
            ),
            node(
                build_bigjoin_node, 
                inputs=["parquet_raw", "params:etl"], 
                outputs="bigjoin",
                name="build_bigjoin",
            ),
            node(
                prune_columns_node,
                inputs=["bigjoin", "params:etl"],
                outputs="bigjoin_clean",
                name="prune_columns",
            ),
            node(
                create_mini_meta_node,
                inputs=["bigjoin_clean", "params:etl"],
                outputs="mini_meta",
                name="create_mini_meta",
            )
        ]
    )