"""Pipeline registry for the Meta Kaggle project."""

from typing import Dict

from kedro.pipeline import Pipeline

from src.pipeline.etl import create_pipeline as create_etl_pipeline
from src.pipeline.features import create_pipeline as create_features_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A dictionary mapping pipeline names to ``Pipeline`` objects.
    """
    etl_pipeline = create_etl_pipeline()
    features_pipeline = create_features_pipeline()

    # Create a sequential pipeline that runs ETL followed by features
    full_pipeline = etl_pipeline + features_pipeline

    return {
        "__default__": full_pipeline,
        "etl": etl_pipeline,
        "features": features_pipeline,
        "full": full_pipeline,
    }