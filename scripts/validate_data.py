#!/usr/bin/env python
"""
Validation and sanity checks for processed data.
"""
import polars as pl
from pathlib import Path
import sys

DATA_DIR = Path("data")
CLEAN_DATA_PATH = DATA_DIR / "intermediate/kernel_bigjoin_clean.parquet"


def validate_data() -> bool:
    """
    Run basic validation checks on the cleaned data.
    
    Returns:
        bool: True if all checks pass, False otherwise
    """
    print(f"Validating data in {CLEAN_DATA_PATH}")
    
    df = pl.scan_parquet(CLEAN_DATA_PATH)
    
    # Check 1: Basic row counts
    row_count = df.select(pl.count()).collect().item()
    print(f"Row count: {row_count:,}")
    if row_count < 100_000:  # Arbitrary threshold for demonstration
        print("ERROR: Row count is suspiciously low!")
        return False
    
    # Check 2: No NA in key foreign keys
    nulls = df.filter(pl.col("comp_id").is_null()).select(pl.count()).collect().item()
    print(f"Null comp_id count: {nulls:,}")
    if nulls > 0:
        print(f"WARNING: Found {nulls} kernels without competition id!")
    
    # Check 3: Date range sanity
    if "kernel_ts" in df.columns:
        dates = df.select(pl.min("kernel_ts"), pl.max("kernel_ts")).collect().rows()
        min_date, max_date = dates[0]
        print(f"Kernels span: {min_date} to {max_date}")
        
    print("All validation checks completed")
    return True


def main() -> None:
    """Run validation checks and exit with appropriate status code."""
    result = validate_data()
    if not result:
        sys.exit(1)


if __name__ == "__main__":
    main()