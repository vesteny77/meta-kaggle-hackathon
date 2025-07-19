#!/usr/bin/env python
"""
Convert CSV files from Meta Kaggle dataset to Parquet format.
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import polars as pl

SRC = Path("data/raw_csv")
DST = Path("data/parquet/raw")


def convert(name: str, date_col: str | None = None) -> tuple[str, float]:
    """
    Convert a single CSV file to Parquet format.

    Args:
        name: The name of the CSV file without extension
        date_col: Optional column to parse as datetime and use for partitioning

    Returns:
        tuple: Table name and processing time in seconds
    """
    start_time = time.time()
    src_file = SRC / f"{name}.csv"

    # Ensure output directory exists
    DST.mkdir(parents=True, exist_ok=True)

    # Use optimized CSV read settings
    df = pl.read_csv(
        src_file,
        infer_schema_length=10000,
        ignore_errors=True,  # cope with ragged rows
        low_memory=False,  # trade memory for speed
        rechunk=True,  # optimize chunk sizes for performance
        n_threads=os.cpu_count(),  # use all available CPU cores
    )

    # Convert date column if specified
    if date_col and date_col in df.columns:
        df = df.with_columns(
            [pl.col(date_col).str.strptime(pl.Datetime, fmt="%Y-%m-%d").alias(date_col)]
        )

    # Partition by year if date_col available
    writer_opts = {}
    if date_col and date_col in df.columns:
        writer_opts = {"partition_by": [date_col]}

    # Write with optimized settings
    df.write_parquet(
        DST / f"{name}.parquet",
        compression="zstd",
        compression_level=3,  # balance between speed and compression ratio
        statistics=True,  # include statistics for faster queries
        use_pyarrow=True,  # use PyArrow for potentially better performance
        **writer_opts,
    )

    elapsed = time.time() - start_time
    return name, elapsed


def main():
    """Process all tables with specified date columns using parallel processing."""
    TABLES = {
        "KernelVersions": "creationDate",
        "Competitions": "endDate",
        "Datasets": "creationDate",
        "Users": "creationDate",
        "ForumMessages": "postDate",
        # Add more tables as needed
    }

    # Create destination directory if it doesn't exist
    DST.mkdir(parents=True, exist_ok=True)

    print(
        f"Starting conversion of {len(TABLES)} tables using {min(os.cpu_count(), len(TABLES))} processes..."
    )
    overall_start = time.time()

    results = []

    # Use process pool for parallel processing
    max_workers = min(os.cpu_count(), len(TABLES))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all conversion jobs
        future_to_table = {
            executor.submit(convert, tbl, date_col): tbl
            for tbl, date_col in TABLES.items()
        }

        # Process results as they complete
        for future in as_completed(future_to_table):
            tbl = future_to_table[future]
            try:
                name, elapsed = future.result()
                print(f"✓ Successfully converted {name} in {elapsed:.2f} seconds")
                results.append((name, elapsed))
            except Exception as e:
                print(f"✗ Error converting {tbl}: {str(e)}", file=sys.stderr)

    # Print summary
    overall_elapsed = time.time() - overall_start
    print(f"\nConversion complete in {overall_elapsed:.2f} seconds")

    if results:
        print("\nSummary:")
        for name, elapsed in sorted(results, key=lambda x: x[1], reverse=True):
            print(f"  {name}: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
