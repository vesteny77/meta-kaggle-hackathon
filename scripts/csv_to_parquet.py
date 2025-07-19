#!/usr/bin/env python
"""
Convert CSV files from Meta Kaggle dataset to Parquet format.
"""
import polars as pl
from pathlib import Path
import sys

SRC = Path("data/raw_csv")
DST = Path("data/parquet/raw")


def convert(name: str, date_col: str | None = None) -> None:
    """
    Convert a single CSV file to Parquet format.
    
    Args:
        name: The name of the CSV file without extension
        date_col: Optional column to parse as datetime and use for partitioning
    """
    src_file = SRC / f"{name}.csv"
    print(f"→ {src_file}")
    
    # streaming=True keeps memory constant
    df = (
        pl.scan_csv(
            src_file,
            infer_schema_length=10000,
            ignore_errors=True,  # cope with ragged rows
        )
        .with_columns([
            pl.col(date_col).str.strptime(pl.Datetime, fmt="%Y-%m-%d").alias(date_col)
            if date_col else pl.lit(None)
        ])
    )
    
    # Partition by year if date_col available
    writer_opts = {}
    if date_col:
        writer_opts = {"partition_by": [date_col]}
    
    DST.mkdir(parents=True, exist_ok=True)
    
    df.collect().write_parquet(
        DST / f"{name}.parquet",
        compression="zstd",
        **writer_opts
    )


def main():
    """Process all tables with specified date columns."""
    TABLES = {
        "KernelVersions": "creationDate",
        "Competitions": "endDate",
        "Datasets": "creationDate",
        "Users": "creationDate",
        "ForumMessages": "postDate",
        # Add more tables as needed
    }

    for tbl, date_col in TABLES.items():
        try:
            convert(tbl, date_col)
            print(f"✓ Successfully converted {tbl}")
        except Exception as e:
            print(f"✗ Error converting {tbl}: {str(e)}", file=sys.stderr)


if __name__ == "__main__":
    main()