#!/usr/bin/env python
"""
Create a small sample (mini-Meta) from the bigjoin for fast iteration.
"""
import polars as pl
import numpy as np
import argparse
from pathlib import Path

DATA_DIR = Path("data")
SRC = DATA_DIR / "intermediate/kernel_bigjoin_clean.parquet"
DST = DATA_DIR / "mini_meta"


def create_mini_meta(sample_frac: float = 0.01) -> None:
    """
    Create a mini-Meta sample dataset.
    
    Args:
        sample_frac: Fraction of data to sample (default: 1%)
    """
    # Create destination directory if it doesn't exist
    DST.mkdir(exist_ok=True, parents=True)
    
    output_file = DST / f"kernel_bigjoin_{int(sample_frac*100)}pct.parquet"
    
    print(f"Creating {sample_frac*100}% sample at {output_file}")
    
    # Use hash-based sampling for consistent results
    scan = pl.scan_parquet(SRC)
    sampled = (
        scan
        .with_columns(
            (pl.col("kernel_id").hash().abs() % 100 < int(sample_frac * 100))
            .alias("_sample")
        )
        .filter(pl.col("_sample") == True)
        .drop("_sample")
        .collect()
    )
    
    # Write to parquet
    sampled.write_parquet(output_file, compression="zstd")
    print(f"Successfully created mini-Meta with {len(sampled)} rows")


def main() -> None:
    """Parse args and run sampling process."""
    parser = argparse.ArgumentParser(description="Create mini-Meta sample dataset")
    parser.add_argument(
        "--frac", type=float, default=0.01, help="Fraction to sample (default: 0.01)"
    )
    args = parser.parse_args()
    
    create_mini_meta(args.frac)


if __name__ == "__main__":
    main()