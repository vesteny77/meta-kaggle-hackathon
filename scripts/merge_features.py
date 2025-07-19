#!/usr/bin/env python
"""
Merge extracted features with topic data and metadata.

This script combines the raw features, topic assignments, and other metadata
to create the final feature dataset for analysis.
"""
import polars as pl
from pathlib import Path
import logging
import argparse
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
FEATURES_PATH = Path("data/intermediate/kernel_features_raw.parquet")
TOPICS_PATH = Path("data/intermediate/markdown_topics.parquet")
BIGJOIN_PATH = Path("data/intermediate/kernel_bigjoin_clean.parquet")
OUTPUT_PATH = Path("data/processed/kernel_features.parquet")


def main():
    """Merge features with topics and metadata."""
    parser = argparse.ArgumentParser(
        description="Merge features with topics and metadata"
    )
    parser.add_argument(
        "--features", 
        type=str, 
        default=str(FEATURES_PATH),
        help="Path to raw kernel features Parquet file"
    )
    parser.add_argument(
        "--topics", 
        type=str, 
        default=str(TOPICS_PATH),
        help="Path to topic assignments Parquet file"
    )
    parser.add_argument(
        "--metadata", 
        type=str, 
        default=str(BIGJOIN_PATH),
        help="Path to kernel metadata Parquet file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=str(OUTPUT_PATH),
        help="Output path for merged features Parquet file"
    )
    parser.add_argument(
        "--keep-embeddings", 
        action="store_true",
        help="Keep markdown embeddings in the output (default: remove them to save space)"
    )
    args = parser.parse_args()
    
    start_time = time.time()
    logger.info("Starting feature merge")
    
    # Load raw features
    logger.info(f"Loading features from {args.features}")
    try:
        features_df = pl.read_parquet(args.features)
        logger.info(f"Loaded {len(features_df)} kernel features")
    except Exception as e:
        logger.error(f"Error loading features: {str(e)}")
        return
    
    # Load topic assignments if available
    topics_df = None
    if Path(args.topics).exists():
        logger.info(f"Loading topics from {args.topics}")
        try:
            topics_df = pl.read_parquet(args.topics)
            logger.info(f"Loaded {len(topics_df)} topic assignments")
        except Exception as e:
            logger.warning(f"Error loading topics: {str(e)}")
    else:
        logger.warning(f"Topics file not found: {args.topics}")
    
    # Load additional metadata if available
    metadata_df = None
    if Path(args.metadata).exists():
        logger.info(f"Loading metadata from {args.metadata}")
        try:
            # Only load relevant columns to save memory
            metadata_df = pl.read_parquet(
                args.metadata,
                columns=["kernel_id", "kernel_ts", "comp_id", "comp_title", "category"]
            )
            logger.info(f"Loaded metadata for {len(metadata_df)} kernels")
        except Exception as e:
            logger.warning(f"Error loading metadata: {str(e)}")
    else:
        logger.warning(f"Metadata file not found: {args.metadata}")
    
    # Merge features with topics
    result_df = features_df
    if topics_df is not None:
        logger.info("Merging with topic assignments")
        result_df = result_df.join(
            topics_df,
            on="kernel_id",
            how="left"
        )
    
    # Merge with metadata
    if metadata_df is not None:
        logger.info("Merging with metadata")
        result_df = result_df.join(
            metadata_df,
            on="kernel_id",
            how="left"
        )
    
    # Remove embeddings to save space (unless requested to keep them)
    if not args.keep_embeddings and "md_embedding" in result_df.columns:
        logger.info("Removing markdown embeddings to save space")
        result_df = result_df.drop("md_embedding")
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save merged features
    logger.info(f"Saving {len(result_df)} merged features to {output_path}")
    result_df.write_parquet(output_path, compression="zstd")
    
    # Print summary
    logger.info("Feature columns in output:")
    for col in result_df.columns:
        logger.info(f"  - {col}")
    
    elapsed = time.time() - start_time
    logger.info(f"Feature merge completed in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()