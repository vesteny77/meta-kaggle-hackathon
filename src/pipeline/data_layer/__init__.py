"""
Data layer pipeline for the Meta Kaggle project.

This pipeline handles the ETL process, including:
- CSV to Parquet conversion
- Building the bigjoin table
- Pruning columns and cleaning data
- Creating the mini-meta sample
- Data validation
"""

from kedro.pipeline import Pipeline, node
from .nodes import (
    csv_to_parquet,
    build_bigjoin,
    prune_columns,
    create_mini_meta,
    validate_data,
)

def create_pipeline(**kwargs) -> Pipeline:
    """Create the data layer pipeline."""
    return Pipeline(
        [
            node(
                csv_to_parquet,
                inputs=["params:data_layer.src_dir", "params:data_layer.dst_dir", "params:data_layer.tables"],
                outputs="raw_parquet_files",
                name="csv_to_parquet",
            ),
            node(
                build_bigjoin,
                inputs=["raw_parquet_files", "params:data_layer.output_dir"],
                outputs="bigjoin",
                name="build_bigjoin",
            ),
            node(
                prune_columns,
                inputs=["bigjoin", "params:data_layer.pruned_output_path"],
                outputs="bigjoin_clean",
                name="prune_columns",
            ),
            node(
                create_mini_meta,
                inputs=["bigjoin_clean", "params:data_layer.sample_frac"],
                outputs="mini_meta",
                name="create_mini_meta",
            ),
            node(
                validate_data,
                inputs="bigjoin_clean",
                outputs=None,
                name="validate_bigjoin",
            ),
        ]
    )


__all__ = ["create_pipeline"]