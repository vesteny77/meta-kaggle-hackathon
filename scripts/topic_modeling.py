#!/usr/bin/env python
"""
Generate topic models from markdown embeddings in kernel features.

This script uses BERTopic to create topic clusters from markdown cell embeddings.
"""
import polars as pl
from pathlib import Path
import numpy as np
import logging
import argparse
import pickle
import time
from bertopic import BERTopic

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
FEATURES_PATH = Path("data/intermediate/kernel_features_raw.parquet")
OUTPUT_DIR = Path("data/intermediate")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    """Generate topic models from markdown embeddings."""
    parser = argparse.ArgumentParser(
        description="Generate topic models from markdown embeddings"
    )
    parser.add_argument(
        "--features", 
        type=str, 
        default=str(FEATURES_PATH),
        help="Path to kernel features Parquet file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=str(OUTPUT_DIR / "markdown_topics.parquet"),
        help="Output path for topic Parquet file"
    )
    parser.add_argument(
        "--model-output", 
        type=str, 
        default=str(OUTPUT_DIR / "topic_model.pkl"),
        help="Output path for topic model pickle file"
    )
    parser.add_argument(
        "--num-topics", 
        type=int, 
        default=None, 
        help="Number of topics (default: auto)"
    )
    args = parser.parse_args()
    
    start_time = time.time()
    logger.info("Starting topic modeling")
    
    # Load features with embeddings
    logger.info(f"Loading features from {args.features}")
    try:
        df = pl.read_parquet(args.features)
        logger.info(f"Loaded {len(df)} kernel features")
    except Exception as e:
        logger.error(f"Error loading features: {str(e)}")
        return
    
    # Filter for kernels with embeddings
    if "md_embedding" not in df.columns:
        logger.error("No markdown embeddings found in features")
        return
    
    embedding_df = df.filter(pl.col("md_embedding").is_not_null())
    kernel_ids = embedding_df["kernel_id"].to_list()
    
    logger.info(f"Found {len(embedding_df)} kernels with markdown embeddings")
    if len(embedding_df) < 100:
        logger.warning("Very few embeddings found, topic modeling may not be effective")
        if len(embedding_df) < 10:
            logger.error("Not enough embeddings for topic modeling")
            return
    
    # Extract embeddings into numpy array
    logger.info("Extracting embeddings")
    embeddings = np.vstack(embedding_df["md_embedding"].to_list())
    logger.info(f"Embeddings shape: {embeddings.shape}")
    
    # Train topic model
    logger.info("Training BERTopic model")
    model_kwargs = {}
    if args.num_topics:
        model_kwargs["nr_topics"] = args.num_topics
        
    topic_model = BERTopic(verbose=True, **model_kwargs)
    topics, probs = topic_model.fit_transform(embeddings)
    
    # Save model
    model_path = Path(args.model_output)
    logger.info(f"Saving model to {model_path}")
    with open(model_path, "wb") as f:
        pickle.dump(topic_model, f)
    
    # Create topic dataframe
    topic_df = pl.DataFrame({
        "kernel_id": kernel_ids,
        "topic_id": topics,
        "topic_prob": probs.max(axis=1) if probs.ndim > 1 else probs
    })
    
    # Add topic labels
    topic_info = topic_model.get_topic_info()
    topic_dict = {row["Topic"]: row["Name"] for _, row in topic_info.iterrows()}
    topic_df = topic_df.with_columns([
        pl.col("topic_id").map_elements(lambda x: topic_dict.get(x, f"Topic {x}")).alias("topic_name")
    ])
    
    # Save topics
    logger.info(f"Saving {len(topic_df)} topic assignments to {args.output}")
    topic_df.write_parquet(args.output)
    
    # Print summary
    topic_counts = topic_df.group_by("topic_name").count().sort("count", descending=True)
    logger.info(f"Top 10 topics:")
    for row in topic_counts.head(10).iter_rows(named=True):
        logger.info(f"  {row['topic_name']}: {row['count']} kernels")
    
    elapsed = time.time() - start_time
    logger.info(f"Topic modeling completed in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()