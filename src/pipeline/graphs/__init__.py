"""
Graph analytics pipeline for the Meta Kaggle project.
"""

from kedro.pipeline import Pipeline

from .nodes import create_pipeline


__all__ = ["create_pipeline"]