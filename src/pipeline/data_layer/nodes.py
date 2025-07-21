"""
Pipeline node function definitions for the data layer.
"""

import sys
from pathlib import Path
import shutil

import duckdb
import polars as pl
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import logging


def csv_to_parquet(src_dir: Path, dst_dir: Path, tables: dict) -> dict:
    """
    Convert CSV files to Parquet format.

    Args:
        src_dir: Source directory for CSV files
        dst_dir: Destination directory for Parquet files
        tables: Dictionary mapping table names to date columns

    Returns:
        dict: Dictionary of output Parquet file paths
    """
    # Ensure source and destination directories are Path objects
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    # Create destination directory if it doesn't exist
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Verify that source directory exists and is actually a directory
    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory {src_dir} does not exist")
    if not src_dir.is_dir():
        raise NotADirectoryError(f"{src_dir} is not a directory")

    outputs = {}

    for tbl, date_col in tables.items():
        try:
            src_file = src_dir / f"{tbl}.csv"
            if not src_file.exists():
                print(f"Warning: {src_file} does not exist", file=sys.stderr)
                continue

            output_path = dst_dir / f"{tbl}.parquet"
            print(f"Converting {tbl} from {src_file} to {output_path}...")

            # Clean up existing output path to prevent IsADirectoryError
            if output_path.exists():
                if output_path.is_dir():
                    shutil.rmtree(output_path)
                else:
                    output_path.unlink()

            # Use Polars to read the CSV and infer schema
            df = pl.read_csv(src_file, infer_schema_length=10000, ignore_errors=True)

            # Special handling for Kernels.csv to clean up CurrentKernelVersionId
            if tbl == "Kernels":
                fix_cols = [
                    "CurrentKernelVersionId",
                    "ForumTopicId",
                ]
                for colname in fix_cols:
                    if colname in df.columns:
                        df = df.with_columns(
                            pl.when(pl.col(colname) == "")
                            .then(None)
                            .otherwise(pl.col(colname))
                            .cast(pl.Int64)
                            .alias(colname)
                        )

            # Special handling for Submissions.csv to clean up SourceKernelVersionId
            if tbl == "Submissions" and "SourceKernelVersionId" in df.columns:
                df = df.with_columns(
                    pl.when(pl.col("SourceKernelVersionId") == "")
                    .then(None)
                    .otherwise(pl.col("SourceKernelVersionId"))
                    .cast(pl.Int64)
                    .alias("SourceKernelVersionId")
                )

            # If a date column is specified but not found in the CSV, skip this table
            if date_col and date_col not in df.columns:
                print(f"Warning: Date column '{date_col}' not found in {tbl}. Skipping conversion of this file.", file=sys.stderr)
                continue

            # If a date column is specified, attempt to parse it
            if date_col:
                # Define date formats to try for parsing
                date_formats = [
                    "%Y-%m-%d",
                    "%Y-%m-%d %H:%M:%S",
                    "%m/%d/%Y",
                    "%m/%d/%Y %H:%M:%S",
                ]

                parsed = False
                for fmt in date_formats:
                    try:
                        # Try to parse the date column with the given format
                        df = df.with_columns(
                            pl.col(date_col).str.strptime(pl.Date, fmt, strict=False)
                        )
                        print(f"Successfully parsed date column '{date_col}' with format '{fmt}'")
                        parsed = True
                        break  # Exit after first successful parse
                    except Exception:
                        continue  # Try the next format

                if not parsed:
                    print(f"Warning: Could not parse date column '{date_col}' in {tbl}. It will be kept as a string.", file=sys.stderr)

            # Write the DataFrame to a Parquet file
            df.write_parquet(output_path, compression="zstd")

            outputs[tbl] = output_path
            print(f"Successfully converted {tbl}")

        except Exception as e:
            print(f"Error converting {tbl}: {str(e)}", file=sys.stderr)
            # Re-raise the exception to halt the pipeline on error
            raise

    return outputs


def build_bigjoin(parquet_files: dict, output_dir: str) -> Path:
    """
    Build the kernel-competition-dataset bigjoin table.

    Args:
        parquet_files: Dictionary of input Parquet file paths
        output_dir: Output directory for the bigjoin

    Returns:
        Path: Path to the output bigjoin Parquet file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "kernel_bigjoin.parquet"

    # Connect to DuckDB
    con = duckdb.connect()

    # Install and load Parquet extension
    con.install_extension("parquet")
    con.load_extension("parquet")

    # Check required tables
    required_tables = [
        "KernelVersions",
        "Competitions",
        "Kernels",
        "KernelVersionCompetitionSources",
        "KernelVersionDatasetSources",
        "Teams",
        "TeamMemberships",
        "Submissions",
        "ForumTopics",
        "Users",
        "KernelAcceleratorTypes",
        "Tags",
        "CompetitionTags"
    ]
    missing_tables = [table for table in required_tables if table not in parquet_files]

    if missing_tables:
        error_msg = f"Missing required tables: {', '.join(missing_tables)}. Please ensure these tables are processed in the CSV to Parquet step."
        print(f"Error: {error_msg}", file=sys.stderr)
        raise ValueError(error_msg)

    # Print information about the tables we're using
    print("\nBuilding bigjoin with the following tables:")
    for table in required_tables:
        if table in parquet_files:
            print(f"- {table}: {parquet_files[table]}")
        else:
            print(f"- {table}: Missing (optional)")
    print()

    def _duckdb_read_path(path_str: str) -> str:
        """Return a glob path if it's a directory, otherwise the original path."""
        p = Path(path_str)
        if p.is_dir():
            return f"{p}/**/*.parquet"
        return path_str

    # Pre-compute competition categories from tags
    con.sql(f"""
    CREATE OR REPLACE TEMP TABLE comp_categories AS
    WITH comp_tags AS (
        SELECT
            ct.CompetitionId as comp_id,
            t.FullPath as tag_path
        FROM read_parquet('{_duckdb_read_path(parquet_files['CompetitionTags'])}') AS ct
        JOIN read_parquet('{_duckdb_read_path(parquet_files['Tags'])}') AS t ON ct.TagId = t.Id
    )
    SELECT
        comp_id,
        split_part(tag_path, ' > ', 2) as category
    FROM comp_tags
    QUALIFY ROW_NUMBER() OVER (PARTITION BY comp_id ORDER BY tag_path) = 1
    """)

    # Build the bigjoin SELECT query (filtered to commit / non-change versions only)
    sql_select = f"""
    SELECT
        kat.Label           AS gpuType,
        k.Id                AS kernel_id,
        kv.Id               AS kernel_version_id,
        kv.Title            AS title,
        kv.CreationDate     AS kernel_ts,
        try_cast(kv.RunningTimeInMilliseconds AS DOUBLE) / 1000.0 AS execSeconds,
        kv.AuthorUserId     AS author_id,
        u.DisplayName       AS user_name,
        u.PerformanceTier   AS user_tier,
        k.TotalVotes        AS total_votes,
        k.TotalViews        AS total_views,
        c.Id                AS comp_id,
        c.Title             AS comp_title,
        cc.category,
        c.EvaluationAlgorithmName AS competition_metric,
        c.EnabledDate       AS comp_enabled_ts,
        kv.IsChange         AS is_commit,
        kvds.SourceDatasetVersionId AS dataset_id,
        t.Id                AS team_id,
        t.TeamName          AS team_name,
        s.PublicScoreFullPrecision AS public_score,
        s.PrivateScoreFullPrecision AS private_score,
        ft.Title            AS forum_topic_title
    FROM read_parquet('{_duckdb_read_path(parquet_files['KernelVersions'])}') AS kv
    LEFT JOIN read_parquet('{_duckdb_read_path(parquet_files['KernelAcceleratorTypes'])}') AS kat ON kv.AcceleratorTypeId = kat.Id
    LEFT JOIN read_parquet('{_duckdb_read_path(parquet_files['Kernels'])}') AS k ON k.CurrentKernelVersionId = kv.Id
    LEFT JOIN read_parquet('{_duckdb_read_path(parquet_files['Users'])}') AS u ON kv.AuthorUserId = u.Id
    LEFT JOIN read_parquet('{_duckdb_read_path(parquet_files['KernelVersionCompetitionSources'])}') AS kvcs ON kvcs.KernelVersionId = kv.Id
    LEFT JOIN read_parquet('{_duckdb_read_path(parquet_files['Competitions'])}') AS c ON c.Id = kvcs.sourceCompetitionId
    LEFT JOIN comp_categories cc ON c.Id = cc.comp_id
    LEFT JOIN read_parquet('{_duckdb_read_path(parquet_files['KernelVersionDatasetSources'])}') AS kvds ON kvds.KernelVersionId = kv.Id
    LEFT JOIN read_parquet('{_duckdb_read_path(parquet_files['TeamMemberships'])}') AS tm ON tm.UserId = kv.AuthorUserId
    LEFT JOIN read_parquet('{_duckdb_read_path(parquet_files['Teams'])}') AS t ON tm.TeamId = t.Id
    LEFT JOIN read_parquet('{_duckdb_read_path(parquet_files['Submissions'])}') AS s ON s.SourceKernelVersionId = kv.Id
    LEFT JOIN read_parquet('{_duckdb_read_path(parquet_files['ForumTopics'])}') AS ft ON k.ForumTopicId = ft.Id
    WHERE kv.IsChange = FALSE
    """

    try:
        # Stream result directly to Parquet without materialising full table in memory
        con.sql(f"COPY ({sql_select}) TO '{output_path}' (FORMAT 'parquet', COMPRESSION 'zstd')")

    except Exception as e:
        print(f"✗ Failed to execute SQL query: {str(e)}", file=sys.stderr)
        # Additional diagnosis info omitted for brevity
        raise

    return output_path


def prune_columns(input_path: Path, output_path: Path) -> Path:
    """Stream-prune columns and fix data types without loading full table in RAM."""

    # Accept either str or Path and normalise to Path objects
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lazy = pl.scan_parquet(input_path)

    # Columns to drop
    cols_to_drop = {"referrerUrl", "isPrivate"}
    keep_cols = [c for c in lazy.columns if c not in cols_to_drop]

    # Transformation pipeline (lazy / streaming)
    lazy = lazy.select(keep_cols)

    if "gpuType" in keep_cols:
        lazy = lazy.with_columns(
            pl.col("gpuType").fill_null("None").cast(pl.Categorical)
        )

    # Use sink_parquet for streaming write (low memory)
    lazy.sink_parquet(output_path, compression="zstd")

    return output_path


# -----------------------
# Mini-meta sampler
# -----------------------

def create_mini_meta(
    input_path: Path, output_path: Path, sample_frac: float = 0.01
) -> Path:
    """
    Create a mini-Meta sample dataset.

    Args:
        input_path: The cleaned bigjoin DataFrame (polars or pandas)
        output_path: Path to output sample file
        sample_frac: Fraction of data to sample (default: 1%)

    Returns:
        Path: Path to the output sample Parquet file
    """

    # Normalise arguments and handle potential mis-ordering (output_path accidentally passed as float)
    # if isinstance(output_path, float) and isinstance(sample_frac, (str, Path)): # The caller likely passed (input_path, sample_frac, output_path)
    #     input_path, output_path, sample_frac = input_path, Path(sample_frac), output_path  # type: ignore[arg-type]

    input_path = Path(input_path)
    output_path = Path("data/mini_meta/kernel_bigjoin_1pct.parquet")

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Hash-based deterministic sampling
    sampled = (
        pl.scan_parquet(input_path)
        .lazy()
        .with_columns(
            (
                pl.col("kernel_version_id").hash().abs() % 100
                < int(sample_frac * 100)
            ).alias("_keep")
        )
        .filter(pl.col("_keep"))
        .drop("_keep")
        .collect()
    )

    # Persist sample
    sampled.write_parquet(output_path, compression="zstd")

    return output_path


def validate_schema(parquet_path: Path, date_col: str = None) -> bool:
    """
    Validate the schema of a Parquet file.

    Args:
        parquet_path: Path to the Parquet file
        date_col: Optional date column to check

    Returns:
        bool: True if validation passes
    """
    # Accept either str or Path and verify existence
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"{parquet_path} does not exist")

    # Load Parquet lazily into DuckDB for inspection
    con = duckdb.connect()
    table_name = parquet_path.stem
    con.register(table_name, pl.scan_parquet(parquet_path))

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

    result = con.sql(query).df()
    row_count = result["rows"].iloc[0]

    # Basic validation - ensure we have rows
    validation_passed = row_count > 0

    # If date column provided, check date range
    if date_col and validation_passed:
        min_date = result["first_date"].iloc[0]
        max_date = result["last_date"].iloc[0]
        # Ensure dates are not null
        validation_passed = min_date is not None and max_date is not None

    # Cast to built-in bool to avoid numpy.bool_ causing identity comparison issues in tests
    return bool(validation_passed)


# -----------------------
# Data validation
# -----------------------

def validate_data(input_obj: "Path | pd.DataFrame | pl.DataFrame") -> bool:
    """
    Run basic validation checks on the cleaned data.

    Args:
        input_obj: Either a Path to the cleaned bigjoin parquet or a DataFrame

    Returns:
        bool: True if all checks pass, False otherwise
    """
    # Convert input into a polars DataFrame
    try:
        if isinstance(input_obj, (str, Path)):
            df = pl.scan_parquet(input_obj)
        elif isinstance(input_obj, pd.DataFrame):
            df = pl.from_pandas(input_obj).lazy()
        else:  # assume polars DataFrame
            df = input_obj.lazy()

        # Basic non-emptiness check
        row_count = df.select(pl.count()).collect().item()
        return row_count > 0
    except Exception:
        return False