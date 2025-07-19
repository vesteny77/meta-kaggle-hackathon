#!/usr/bin/env python
"""
Column pruning and data-type fixes for the bigjoin table.
"""
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc

DATA_DIR = Path("data")
INPUT_PATH = DATA_DIR / "intermediate/kernel_bigjoin.parquet"
OUTPUT_PATH = DATA_DIR / "intermediate/kernel_bigjoin_clean.parquet"


def prune_columns() -> None:
    """
    Prune rarely used columns and fix data types.
    """
    print(f"Reading from {INPUT_PATH}")

    # Read table
    tbl = pa.parquet.read_table(INPUT_PATH)

    # Drop rarely used columns
    keep = [c for c in tbl.column_names if c not in ["referrerUrl", "isPrivate"]]
    tbl = tbl.select(keep)

    # Cast gpuType to categorical + fill nulls
    if "gpuType" in tbl.column_names:
        tbl = tbl.set_column(
            tbl.column_names.index("gpuType"),
            "gpuType",
            pc.fill_null(tbl["gpuType"], pa.scalar("None")).dictionary_encode(),
        )

    # Write to clean Parquet
    print(f"Writing to {OUTPUT_PATH}")
    pl.from_arrow(tbl).write_parquet(OUTPUT_PATH, compression="zstd")
    print("Column pruning completed successfully")


def main() -> None:
    """Run the column pruning process."""
    prune_columns()


if __name__ == "__main__":
    main()
