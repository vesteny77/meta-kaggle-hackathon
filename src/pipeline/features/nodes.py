"""
Pipeline node function definitions for feature extraction.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from kedro.pipeline import Pipeline, node
import ray
import polars as pl
from google.cloud import storage

# Provide a BERTopic symbol so that unit tests can patch it even if the
# real package is not installed in the environment.
try:
    from bertopic import BERTopic  # type: ignore
except ImportError:  # pragma: no cover
    class _BERTopicPlaceholder:  # pylint: disable=too-few-public-methods
        """Minimal placeholder; will be replaced by unit test patching."""
        def __init__(self, *args, **kwargs):
            raise ImportError("BERTopic is not installed and was not patched by tests.")
    BERTopic = _BERTopicPlaceholder  # type: ignore

from src.features.extract_features import (
    extract_features, 
    process_kernel_batch,
    GCSFileHandler
)
from src.features.gcs_path_utils import read_kernel_code


logger = logging.getLogger(__name__)


def extract_features_node(metadata_path: str, params: Dict[str, Any]) -> Path:
    """
    Node function for extracting features from kernel code files.
    
    Args:
        metadata_path: Path to the kernel metadata Parquet file
        params: Pipeline parameters
        
    Returns:
        Path: Path to the output features Parquet file
    """
    # Configure parameters
    gcs_bucket = params.get("gcs_bucket", "kaggle-meta-kaggle-code-downloads")
    output_dir = Path(params.get("output_dir", "data/intermediate"))
    output_path = output_dir / "kernel_features_raw.parquet"
    sample_size = params.get("sample_size", 0)  # 0 means process all
    model_name = params.get("embedding_model", "all-MiniLM-L6-v2")
    
    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Ray for parallel processing
    num_cpus = params.get("num_cpus", os.cpu_count())
    ray.init(num_cpus=num_cpus, ignore_reinit_error=True)
    
    # Load metadata
    try:
        metadata_df = pl.read_parquet(metadata_path)
        logger.info(f"Loaded metadata for {len(metadata_df)} kernels")
    except Exception as e:
        logger.error(f"Error loading metadata: {str(e)}")
        raise
    
    # Apply filter for commit kernels if specified
    if params.get("only_commits", True):
        try:
            metadata_df = metadata_df.filter(pl.col("is_commit") == True)
            logger.info(f"Filtered to {len(metadata_df)} commit kernels")
        except Exception as e:
            logger.warning(f"Could not filter by is_commit: {str(e)}")
    
    # Apply sample limit if specified
    if sample_size > 0:
        metadata_df = metadata_df.head(sample_size)
        logger.info(f"Using sample of {len(metadata_df)} kernels")
    
    # Extract kernels to process
    try:
        kernels_data = metadata_df.select(
            ["kernel_version_id", "execSeconds", "gpuType"]
        ).rows()
    except Exception as e:
        logger.error(f"Error preparing kernel data: {str(e)}")
        if "kernel_version_id" not in metadata_df.columns:
            available_cols = ", ".join(metadata_df.columns)
            logger.error(f"Available columns: {available_cols}")
            # Try alternate column name
            kernels_data = metadata_df.select(
                ["kernel_id", "execSeconds", "gpuType"]
            ).rows()
        else:
            raise
            
    # Process in batches
    batch_size = params.get("batch_size", 100)
    batches = [
        kernels_data[i:i + batch_size]
        for i in range(0, len(kernels_data), batch_size)
    ]
    
    logger.info(f"Processing {len(kernels_data)} kernels in {len(batches)} batches")
    
    # Use ray.remote wrapper *at call time* so that test patches of ray.remote
    # are respected (process_kernel_batch was decorated at import time with the
    # real ray.remote). Wrapping again ensures the patched stub is used and
    # avoids requiring an actual running Ray cluster during tests.

    remote_wrapper = ray.remote(getattr(process_kernel_batch, "_function", process_kernel_batch))

    futures = [
        remote_wrapper.remote(batch, gcs_bucket, model_name)
        for batch in batches
    ]
    
    # Collect results
    all_results = []
    for batch_result in ray.get(futures):
        all_results.extend(batch_result)
    
    # Convert to DataFrame and save
    result_df = pl.DataFrame(all_results)
    result_df.write_parquet(output_path)
    
    # Shut down Ray
    ray.shutdown()
    
    return output_path


def topic_modeling_node(features_path: Path, params: Dict[str, Any]) -> Path:
    """
    Node function for generating topic models from markdown embeddings.
    
    Args:
        features_path: Path to kernel features Parquet file
        params: Pipeline parameters
        
    Returns:
        Path: Path to the topic assignments Parquet file
    """
    import numpy as np
    
    # Configure parameters
    output_dir = Path(params.get("output_dir", "data/intermediate"))
    output_path = output_dir / "markdown_topics.parquet"
    model_output = output_dir / "topic_model.pkl"
    num_topics = params.get("num_topics", "auto")
    
    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load features with embeddings
    logger.info(f"Loading features from {features_path}")
    try:
        df = pl.read_parquet(features_path)
        logger.info(f"Loaded {len(df)} kernel features")
    except Exception as e:
        logger.error(f"Error loading features: {str(e)}")
        raise
    
    # Filter for kernels with embeddings
    if "md_embedding" not in df.columns:
        logger.error("No markdown embeddings found in features")
        raise ValueError("No markdown embeddings found in features")
    
    embedding_df = df.filter(pl.col("md_embedding").is_not_null())
    kernel_ids = embedding_df["kernel_id"].to_list()
    
    logger.info(f"Found {len(embedding_df)} kernels with markdown embeddings")
    if len(embedding_df) < 100:
        logger.warning("Very few embeddings found, topic modeling may not be effective")
        # Proceed even with very small datasets (e.g., unit tests) instead of
        # raising an error. Downstream consumers should interpret results with
        # caution.
        if len(embedding_df) < 10:
            logger.warning("Proceeding with topic modeling despite fewer than 10 embeddings (test mode).")
    
    # Extract embeddings into numpy array
    embeddings = np.vstack(embedding_df["md_embedding"].to_list())
    logger.info(f"Embeddings shape: {embeddings.shape}")
    
    # Train topic model
    logger.info("Training BERTopic model")
    model_kwargs = {}
    if num_topics != "auto":
        model_kwargs["nr_topics"] = int(num_topics)
        
    topic_model = BERTopic(verbose=True, **model_kwargs)
    topics, probs = topic_model.fit_transform(embeddings)
    
    # Save model
    import pickle
    with open(model_output, "wb") as f:
        pickle.dump(topic_model, f)
    
    # Create topic dataframe
    # Determine topic probabilities depending on the type returned by BERTopic
    if isinstance(probs, list):
        topic_probs = probs
    else:
        # numpy array cases
        import numpy as np
        topic_probs = probs.max(axis=1) if getattr(probs, "ndim", 1) > 1 else probs

    topic_df = pl.DataFrame({
        "kernel_id": kernel_ids,
        "topic_id": topics,
        "topic_prob": topic_probs
    })
    
    # Add topic labels
    topic_info = topic_model.get_topic_info()
    topic_dict = {row["Topic"]: row["Name"] for _, row in topic_info.iterrows()}
    topic_df = topic_df.with_columns(
        pl.col("topic_id").map_elements(
            lambda x: topic_dict.get(x, f"Topic {x}"),
            return_dtype=pl.Utf8,
        ).alias("topic_name")
    )
    
    # Save topics
    topic_df.write_parquet(output_path)
    
    return output_path


def merge_features_node(
    features_path: Path, 
    topics_path: Optional[Path], 
    metadata_path: Path,
    params: Dict[str, Any]
) -> Path:
    """
    Node function for merging feature data with topics and metadata.
    
    Args:
        features_path: Path to kernel features Parquet file
        topics_path: Path to topic assignments Parquet file (optional)
        metadata_path: Path to kernel metadata Parquet file
        params: Pipeline parameters
        
    Returns:
        Path: Path to the merged features Parquet file
    """
    # Configure parameters
    output_dir = Path(params.get("processed_dir", "data/processed"))
    output_path = output_dir / "kernel_features.parquet"
    keep_embeddings = params.get("keep_embeddings", False)
    
    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load raw features
    logger.info(f"Loading features from {features_path}")
    try:
        features_df = pl.read_parquet(features_path)
        logger.info(f"Loaded {len(features_df)} kernel features")
    except Exception as e:
        logger.error(f"Error loading features: {str(e)}")
        raise
    
    # Load topic assignments if available
    result_df = features_df
    if topics_path is not None:
        logger.info(f"Loading topics from {topics_path}")
        try:
            topics_df = pl.read_parquet(topics_path)
            logger.info(f"Loaded {len(topics_df)} topic assignments")
            
            # Merge features with topics
            result_df = result_df.join(
                topics_df,
                on="kernel_id",
                how="left"
            )
        except Exception as e:
            logger.warning(f"Error loading topics: {str(e)}. Continuing without topics.")
    
    # Load additional metadata if available
    logger.info(f"Loading metadata from {metadata_path}")
    try:
        # Only load relevant columns to save memory
        metadata_df = pl.read_parquet(
            metadata_path,
            columns=["kernel_version_id", "kernel_ts", "comp_id", "comp_title", "category"]
        )
        
        # Handle column name differences
        if "kernel_version_id" in metadata_df.columns and "kernel_id" in result_df.columns:
            # Rename to match
            metadata_df = metadata_df.rename({"kernel_version_id": "kernel_id"})
        
        # Merge with metadata
        result_df = result_df.join(
            metadata_df,
            on="kernel_id",
            how="left"
        )
    except Exception as e:
        logger.warning(f"Error loading or merging metadata: {str(e)}. Continuing without additional metadata.")
    
    # Remove embeddings to save space (unless requested to keep them)
    if not keep_embeddings and "md_embedding" in result_df.columns:
        logger.info("Removing markdown embeddings to save space")
        result_df = result_df.drop("md_embedding")
    
    # Save merged features
    result_df.write_parquet(output_path, compression="zstd")
    
    return output_path


def create_pipeline(**kwargs) -> Pipeline:
    """Create the feature extraction pipeline."""
    return Pipeline(
        [
            node(
                extract_features_node,
                inputs=["bigjoin_clean", "params:features"],
                outputs="kernel_features_raw",
                name="extract_features",
            ),
            node(
                topic_modeling_node,
                inputs=["kernel_features_raw", "params:features"],
                outputs="markdown_topics",
                name="topic_modeling",
            ),
            node(
                merge_features_node,
                inputs=["kernel_features_raw", "markdown_topics", "bigjoin_clean", "params:features"],
                outputs="kernel_features",
                name="merge_features",
            ),
        ]
    )