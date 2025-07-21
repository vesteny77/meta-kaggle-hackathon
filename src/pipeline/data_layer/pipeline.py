"""
ETL pipeline for the Meta Kaggle project.
"""

from pathlib import Path

from kedro.pipeline import Pipeline, node

from .nodes import (
    build_bigjoin,
    create_mini_meta,
    csv_to_parquet,
    prune_columns,
    validate_data,
    validate_schema,
)

import logging


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
    tables = params.get(
        "tables",
        {
            "KernelVersions": "CreationDate",
            "Competitions": "DeadlineDate",
            "Datasets": "CreationDate",
            "Kernels": "",
            "KernelVersionCompetitionSources": "",
            "KernelVersionDatasetSources": "",
            "Users": "RegisterDate",
            "ForumMessages": "PostDate",
        },
    )

    return csv_to_parquet(src_dir, dst_dir, tables)


def build_bigjoin_node(parquet_raw, params):
    """
    Node function wrapper for build_bigjoin.

    Args:
        parquet_raw: Dictionary of input Parquet file paths
        params: Pipeline parameters

    Returns:
        pl.DataFrame: The bigjoined polars DataFrame
    """
    output_dir = Path(params.get("intermediate_dir", "data/intermediate"))
    return build_bigjoin(parquet_raw, output_dir)


def prune_columns_node(bigjoin_path, params):
    """
    Node function wrapper for prune_columns.

    Args:
        bigjoin_path: The bigjoined polars DataFrame
        params: Pipeline parameters

    Returns:
        Path: Path to the output cleaned Parquet file
    """
    intermediate_dir = Path(params.get("intermediate_dir", "data/intermediate"))
    output_path = intermediate_dir / "kernel_bigjoin_clean.parquet"
    return prune_columns(bigjoin_path, output_path)


def validate_schema_node(parquet_raw, params):
    """
    Node function wrapper for validate_schema.

    Args:
        parquet_raw: Dictionary of input Parquet file paths
        params: Pipeline parameters

    Returns:
        bool: True if validation passes
    """
    # Validate the KernelVersions file, which is crucial for the pipeline
    kernel_versions_path = parquet_raw.get("KernelVersions", None)
    date_col = "creationDate"

    if kernel_versions_path:
        return validate_schema(Path(kernel_versions_path), date_col)
    return False


def validate_data_node(bigjoin_clean, params):
    """
    Node function wrapper for validate_data.

    Args:
        bigjoin_clean: Path to the cleaned bigjoin file
        params: Pipeline parameters

    Returns:
        bool: True if all checks pass
    """
    return validate_data(bigjoin_clean)


def create_mini_meta_node(bigjoin_clean_path, params):
    """
    Node function wrapper for create_mini_meta.

    Args:
        bigjoin_clean_path: The cleaned bigjoin DataFrame
        params: Pipeline parameters

    Returns:
        Path: Path to the mini-meta sample parquet
    """
    sample_frac = params.get("sample_frac", 0.01)
    mini_meta_dir = Path("data/mini_meta")
    output_path = mini_meta_dir / f"kernel_bigjoin_{int(sample_frac*100)}pct.parquet"
    logging.info(f"mini_meta_dir: {mini_meta_dir}")
    logging.info(f"output_path: {output_path}")
    return create_mini_meta(input_path=bigjoin_clean_path, output_path=output_path, sample_frac=sample_frac)


def create_pipeline(**kwargs):
    """Create the ETL pipeline."""
    return Pipeline(
        [
            node(
                csv_to_parquet_node,
                inputs="params:data_layer",
                outputs="parquet_raw",
                name="csv_to_parquet",
            ),
            node(
                validate_schema_node,
                inputs=["parquet_raw", "params:data_layer"],
                outputs="schema_validation",
                name="validate_schema",
            ),
            node(
                build_bigjoin_node,
                inputs=["parquet_raw", "params:data_layer"],
                outputs="bigjoin",
                name="build_bigjoin",
            ),
            node(
                prune_columns_node,
                inputs=["bigjoin", "params:data_layer"],
                outputs="bigjoin_clean",
                name="prune_columns",
            ),
            node(
                validate_data_node,
                inputs=["bigjoin_clean", "params:data_layer"],
                outputs="data_validation",
                name="validate_data",
            ),
            node(
                create_mini_meta_node,
                inputs=["bigjoin_clean", "params:data_layer"],
                outputs="mini_meta",
                name="create_mini_meta",
            ),
        ]
    )