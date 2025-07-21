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
)
from src.features.local_path_utils import read_kernel_code


logger = logging.getLogger(__name__)


def extract_features_node(params: Dict[str, Any]) -> Path:
    """Efficiently extract raw kernel-level features without exhausting RAM.

    The function now:
    1. Streams metadata in *batches* instead of loading everything at once.
    2. Sends each batch to a Ray worker which returns a Python list of dicts.
    3. Immediately persists each batch to its own Parquet shard, then frees memory.
    The result is a folder full of small Parquet files that together make up
    ``data/intermediate/kernel_features_raw/``. Down-stream Polars / DuckDB can
    read the folder lazily, so no concatenation step is needed.
    """

    # ------------------------------------------------------------------
    # Configuration & paths
    # ------------------------------------------------------------------
    local_root = Path(params.get("local_code_root", "data/raw_code"))
    output_dir = Path(params.get("output_dir", "data/intermediate")) / "kernel_features_raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = Path(params.get("metadata_path", "data/intermediate/kernel_bigjoin_clean.parquet"))

    model_name = params.get("embedding_model", "all-MiniLM-L6-v2")
    batch_size = params.get("batch_size", 400)              # kernels per Ray task
    sample_size = params.get("sample_size", 0)              # 0 = full dataset
    num_cpus    = params.get("num_cpus", os.cpu_count())
    # Pick a short absolute path for Ray object spill directory to avoid UNIX socket length limits
    default_spill = Path("/tmp/ray_tmp")
    spill_dir   = Path(params.get("spill_dir", default_spill)).resolve()
    spill_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Using Ray spill directory: %s", spill_dir)

    # ------------------------------------------------------------------
    # Initialise Ray with spilling to disk
    # ------------------------------------------------------------------
    ray.shutdown()
    ray.init(
        num_cpus=num_cpus,
        _temp_dir=str(spill_dir),
        include_dashboard=False,
        ignore_reinit_error=True,
        local_mode=True,
    )

    logger.info("Metadata source: %s", metadata_path)

    # ------------------------------------------------------------------
    # Lazy metadata scan
    # ------------------------------------------------------------------
    scan = pl.scan_parquet(metadata_path)

    # Keep only commit kernels if requested (default True)
    # if params.get("only_commits", True):
    #     scan = scan.filter(pl.col("is_commit") == True)

    # Select minimal columns
    columns = [c for c in ["kernel_version_id", "execSeconds", "gpuType"] if c in scan.collect_schema().names()]
    scan = scan.select(columns)

    # Row count (cheap – metadata file is indexed)
    total_rows = scan.select(pl.count()).collect().item()
    if sample_size and sample_size > 0:
        total_rows = min(total_rows, sample_size)
        scan = scan.head(sample_size)
        logger.info("Sample size activated: %d rows", total_rows)
    else:
        logger.info("Processing full dataset: %d rows", total_rows)

    # Ray remote wrapper (patched easily in unit tests)
    remote_wrapper = ray.remote(getattr(process_kernel_batch, "_function", process_kernel_batch))

    # ------------------------------------------------------------------
    # Stream over the scan in windows to keep memory constant
    # ------------------------------------------------------------------
    futures = []
    shard_index = 0
    for offset in range(0, total_rows, batch_size):
        length = min(batch_size, total_rows - offset)
        batch_df = scan.slice(offset, length).collect(streaming=True)

        # Fill missing columns if any
        if "execSeconds" not in batch_df.columns:
            batch_df = batch_df.with_columns(pl.lit(None).alias("execSeconds"))
        if "gpuType" not in batch_df.columns:
            batch_df = batch_df.with_columns(pl.lit("None").alias("gpuType"))

        batch_rows = batch_df.rows()

        futures.append(
            (shard_index, remote_wrapper.remote(batch_rows, str(local_root), model_name))
        )
        shard_index += 1

    logger.info("Submitted %d Ray tasks", len(futures))

    # ------------------------------------------------------------------
    # As each task finishes, write its shard and free memory
    # ------------------------------------------------------------------
    for shard_idx, future in futures:
        batch_result = ray.get(future)
        if not batch_result:
            continue
        shard_path = output_dir / f"features_{shard_idx:06}.parquet"
        pl.DataFrame(batch_result).write_parquet(shard_path, compression="zstd")
        logger.info("Shard %s written (%d rows)", shard_path.name, len(batch_result))

    ray.shutdown()

    # Return folder path; Kedro catalog can point to it as a partitioned dataset
    return output_dir


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
        fp = Path(features_path)
        if fp.is_dir():
            # Folder of Parquet shards written by extract_features_node
            df = pl.scan_parquet(str(fp / "*.parquet")).collect()
        else:
            df = pl.read_parquet(fp)
        logger.info("Loaded %d kernel features", len(df))
    except Exception as e:
        logger.error("Error loading features: %s", e)
        raise
    
    # Filter for kernels with embeddings
    if "md_embedding" not in df.columns or "md_text" not in df.columns:
        logger.error("No markdown embeddings found in features")
        raise ValueError("No markdown embeddings found in features")
    
    embedding_df = df.filter(pl.col("md_embedding").is_not_null())
    kernel_ids = embedding_df["kernel_id"].to_list()
    documents = embedding_df["md_text"].to_list()
    
    emb_count = len(embedding_df)
    logger.info(f"Found {emb_count} kernels with markdown embeddings")

    MIN_DOCS = 10
    if emb_count < MIN_DOCS:
        logger.warning(
            "Only %d markdown embeddings found (<%d). Skipping topic modeling step.",
            emb_count,
            MIN_DOCS,
        )
        return None

    if emb_count < 100:
        logger.warning("Very few embeddings found, topic modeling may not be effective")
    
    # Extract embeddings into numpy array
    embeddings = np.vstack(embedding_df["md_embedding"].to_list())
    logger.info(f"Embeddings shape: {embeddings.shape}")
    
    # Train topic model using embeddings; disable probability calculation for speed
    logger.info("Training BERTopic model (embeddings only)")
    model_kwargs = {"calculate_probabilities": False}
    if num_topics != "auto":
        model_kwargs["nr_topics"] = int(num_topics)

    topic_model = BERTopic(verbose=True, **model_kwargs)

    logger.info("Calling BERTopic.fit_transform on %d docs", len(kernel_ids))
    try:
        topics, _ = topic_model.fit_transform(documents, embeddings)
    except Exception as e:
        logger.exception("BERTopic fit_transform failed: %s", e)
        raise

    topic_df = pl.DataFrame({
        "kernel_id": kernel_ids,
        "topic_id": topics,
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
    params: Dict[str, Any],
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
        fp = Path(features_path)
        if fp.is_dir():
            features_df = pl.scan_parquet(str(fp / "*.parquet")).collect()
        else:
            features_df = pl.read_parquet(fp)
        logger.info("Loaded %d kernel features", len(features_df))
    except Exception as e:
        logger.error("Error loading features: %s", e)
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
    
    metadata_path = Path(params.get("metadata_path", "data/intermediate/kernel_bigjoin_clean.parquet"))
    
    # Select only the necessary columns from metadata to merge
    metadata_cols = [
        "kernel_version_id", "kernel_ts", "comp_id", 
        "comp_title", "team_id", "user_tier"
    ]
    
    # Check which of these are actually available
    meta_schema = pl.scan_parquet(metadata_path).collect_schema().names()
    available_meta_cols = [c for c in metadata_cols if c in meta_schema]
    
    metadata_df = pl.read_parquet(metadata_path, columns=available_meta_cols)

    # Rename kernel_version_id to kernel_id for joining
    if "kernel_version_id" in metadata_df.columns:
        metadata_df = metadata_df.rename({"kernel_version_id": "kernel_id"})

    # Join with the features
    result_df = result_df.join(metadata_df, on="kernel_id", how="left")
    
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
                inputs=["params:features"],
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
                inputs=["kernel_features_raw", "markdown_topics", "params:features"],
                outputs="kernel_features",
                name="merge_features",
            ),
        ]
    )