"""Pipeline registry for the Meta Kaggle project."""
from typing import Dict

from kedro.pipeline import Pipeline

from src.pipeline.etl import create_pipeline as create_etl_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A dictionary mapping pipeline names to ``Pipeline`` objects.
    """
    etl_pipeline = create_etl_pipeline()
    
    return {
        "__default__": etl_pipeline,
        "etl": etl_pipeline,
    }