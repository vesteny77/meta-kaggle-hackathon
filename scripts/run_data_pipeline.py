#!/usr/bin/env python
"""
Main entry point for running the data preparation pipeline.
"""
import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str) -> bool:
    """
    Run a shell command and handle errors.
    
    Args:
        cmd: Command to run
        description: Description of the command
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n==== {description} ====")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"ERROR: Failed to {description}", file=sys.stderr)
        return False
    return True


def main() -> None:
    """Run the entire data preparation pipeline."""
    steps = [
        ("python scripts/csv_to_parquet.py", "Convert CSV to Parquet"),
        ("python scripts/validate_schema.py data/parquet/raw/KernelVersions.parquet creationDate", 
         "Validate KernelVersions schema"),
        ("python scripts/build_bigjoin.py", "Build kernel bigjoin"),
        ("python scripts/prune_columns.py", "Prune columns and fix data types"),
        ("python scripts/validate_data.py", "Validate processed data"),
        ("python scripts/make_mini_meta.py --frac 0.01", "Create 1% mini-Meta sample"),
    ]
    
    success = True
    for cmd, desc in steps:
        if not run_command(cmd, desc):
            success = False
            break
    
    if success:
        print("\n✅ Data pipeline completed successfully!")
    else:
        print("\n❌ Data pipeline failed!", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()