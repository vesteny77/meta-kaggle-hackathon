"""Project pipelines."""

from typing import Dict

from kedro.pipeline import Pipeline
from src.pipeline.data_layer import create_pipeline as create_data_layer_pipeline
from src.pipeline.features import create_pipeline as create_feat_pipeline
from src.pipeline.graphs import create_pipeline as create_graphs_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines."""
    data_layer_pipeline = create_data_layer_pipeline()
    feat_pipeline = create_feat_pipeline()
    graphs_pipeline = create_graphs_pipeline()

    return {
        "__default__": data_layer_pipeline,
        "data_layer": data_layer_pipeline,
        "features": feat_pipeline,
        "graphs": graphs_pipeline,
    }
