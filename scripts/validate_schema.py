#!/usr/bin/env python
"""
Quick schema validation script for Parquet files.
"""
import sys
from pathlib import Path

import duckdb
import polars as pl


def validate_schema(file_path: str, date_col: str | None = None) -> None:
    """
    Validate the schema of a Parquet file.

    Args:
        file_path: Path to the Parquet file
        date_col: Optional date column to check
    """
    print(f"Validating {file_path}...")

    # Load Parquet lazily into DuckDB for inspection
    con = duckdb.connect()
    table_name = Path(file_path).stem
    con.register(table_name, pl.scan_parquet(file_path))

    # Check row count and date range
    query = (
        f"""
        SELECT
            COUNT(*) AS rows,
            MIN({date_col}) AS first_date,
            MAX({date_col}) AS last_date
        FROM {table_name}
    """
        if date_col
        else f"""
        SELECT COUNT(*) AS rows
        FROM {table_name}
    """
    )

    result = con.sql(query).fetch_df()
    print(result)


def main():
    """Validate schemas for key tables."""
    if len(sys.argv) < 2:
        print("Usage: python validate_schema.py <parquet_file> [date_column]")
        sys.exit(1)

    file_path = sys.argv[1]
    date_col = sys.argv[2] if len(sys.argv) > 2 else None

    validate_schema(file_path, date_col)


if __name__ == "__main__":
    main()
