#!/usr/bin/env python
"""
Heavy joins with DuckDB-Python to create analysis-ready tables.
"""
from pathlib import Path

import duckdb
import polars as pl

DATA_DIR = Path("data")
PARQUET_DIR = DATA_DIR / "parquet/raw"
OUTPUT_DIR = DATA_DIR / "intermediate"


def build_kernel_bigjoin() -> None:
    """
    Build the kernel-competition-dataset bigjoin table.
    """
    print("Building kernel-competition-dataset bigjoin...")

    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Connect to DuckDB
    con = duckdb.connect()

    # Install and load Parquet extension
    con.install_extension("parquet")
    con.load_extension("parquet")

    # Register Parquet scans (zero-copy)
    con.register("k", pl.scan_parquet(PARQUET_DIR / "KernelVersions.parquet"))
    con.register("c", pl.scan_parquet(PARQUET_DIR / "Competitions.parquet"))
    con.register(
        "kvds", pl.scan_parquet(PARQUET_DIR / "KernelVersionDatasetSources.parquet")
    )
    con.register(
        "kvcs", pl.scan_parquet(PARQUET_DIR / "KernelVersionCompetitionSources.parquet")
    )

    # Build the bigjoin SQL
    sql = """
    CREATE OR REPLACE TABLE kernel_bigjoin AS
    SELECT
        k.id               AS kernel_id,
        k.creationDate     AS kernel_ts,
        k.authorUserId     AS author_id,
        k.totalTokens,
        k.execSeconds,
        k.gpuType,
        k.isCommit,
        c.id               AS comp_id,
        c.title            AS comp_title,
        c.category,
        c.endDate,
        kvds.datasetId     AS dataset_id
    FROM k
    LEFT JOIN kvcs ON kvcs.kernelVersionId = k.id
    LEFT JOIN c ON kvcs.sourceCompetitionId = c.id
    LEFT JOIN kvds ON kvds.kernelVersionId = k.id
    WHERE k.isCommit = TRUE
    """
    con.sql(sql)

    # Persist to Parquet (partition by comp_year)
    export_sql = """
    COPY kernel_bigjoin
    TO '{output_path}'
       (FORMAT 'parquet', COMPRESSION 'zstd',
        PARTITION_BY (strftime(endDate, '%Y')))
    """.format(
        output_path=str(OUTPUT_DIR / "kernel_bigjoin.parquet")
    )

    con.sql(export_sql)
    print("Successfully created kernel_bigjoin.parquet")


def main() -> None:
    """Run the bigjoin process."""
    build_kernel_bigjoin()


if __name__ == "__main__":
    main()
