"""
Pipeline node function definitions.
"""

import sys
from pathlib import Path

import duckdb
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


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

    # Common date formats to try
    date_formats = [
        "%Y-%m-%d",  # 2023-01-31
        "%Y-%m-%d %H:%M:%S",  # 2023-01-31 14:30:45
        "%m/%d/%Y",  # 01/31/2023
        "%m/%d/%Y %H:%M:%S",  # 01/31/2023 14:30:45
    ]

    # Set chunking parameters based on file size
    # Files larger than this will be processed in chunks
    LARGE_FILE_SIZE = 1 * 1024 * 1024 * 1024  # 1 GB
    DEFAULT_CHUNK_SIZE = 500000  # default rows per chunk

    # Try to determine available system memory and adjust chunk size accordingly
    try:
        import psutil

        available_memory = psutil.virtual_memory().available
        # Use at most 25% of available memory as a safety margin
        safe_memory = available_memory * 0.25
        # Estimate memory per row (very rough estimate: 1000 bytes per row)
        memory_per_row = 1000  # bytes
        # Calculate safe chunk size based on available memory
        memory_based_chunk_size = int(safe_memory / memory_per_row)
        # Use the smaller of default or memory-based chunk size
        CHUNK_SIZE = min(DEFAULT_CHUNK_SIZE, memory_based_chunk_size)
        print(
            f"Available memory: {available_memory / (1024*1024*1024):.2f} GB, using chunk size: {CHUNK_SIZE:,} rows"
        )
    except ImportError:
        # If psutil isn't available, use default chunk size
        CHUNK_SIZE = DEFAULT_CHUNK_SIZE
        print(
            f"Using default chunk size: {CHUNK_SIZE:,} rows (install psutil for adaptive sizing)"
        )
    except Exception as e:
        # If anything goes wrong, fall back to default
        CHUNK_SIZE = DEFAULT_CHUNK_SIZE
        print(
            f"Error determining memory: {str(e)}. Using default chunk size: {CHUNK_SIZE:,} rows"
        )

    for tbl, date_col in tables.items():
        try:
            src_file = src_dir / f"{tbl}.csv"
            if not src_file.exists():
                print(f"Warning: {src_file} does not exist", file=sys.stderr)
                continue

            if not src_file.is_file():
                print(f"Warning: {src_file} exists but is not a file", file=sys.stderr)
                continue

            # Prepare output path and ensure parent directory exists
            output_path = dst_dir / f"{tbl}.parquet"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Check file size to determine if we need chunking
            file_size = src_file.stat().st_size
            use_chunking = file_size > LARGE_FILE_SIZE

            if use_chunking:
                print(
                    f"Large file detected for {tbl} ({file_size / (1024*1024*1024):.2f} GB). Using chunked processing."
                )
        except Exception as e:
            print(f"Error preparing to convert {tbl}: {str(e)}", file=sys.stderr)
            continue

        try:
            print(f"Converting {tbl}...")

            # First, analyze a small sample to determine schema and date format
            sample_scan = pl.scan_csv(
                src_file, infer_schema_length=10000, ignore_errors=True
            )

            # Get a small sample to detect column names and date format
            sample_df = sample_scan.limit(10).collect()

            # Check if date column exists and determine format
            success = False
            fmt_to_use = None

            if date_col:
                try:
                    # Check if column exists with expected case
                    if date_col in sample_df.columns:
                        # Column exists with exact case match
                        error_messages = []

                        # Try each format until one works
                        if not sample_df[date_col].is_empty():
                            for fmt in date_formats:
                                try:
                                    # Attempt to parse with this format
                                    test_df = sample_df.with_columns(
                                        [
                                            pl.col(date_col)
                                            .str.strptime(pl.Datetime, fmt)
                                            .alias(f"{date_col}_parsed")
                                        ]
                                    )
                                    # If we get here, the format works
                                    success = True
                                    fmt_to_use = fmt
                                    print(
                                        f"\u2713 Using date format '{fmt}' for column '{date_col}'"
                                    )
                                    break
                                except Exception as e:
                                    error_messages.append(f"Format '{fmt}': {str(e)}")

                            if not success:
                                # If all formats fail, keep the column as string
                                print(
                                    f"Warning: Could not parse date column '{date_col}' in {tbl}. Keeping as string.",
                                    file=sys.stderr,
                                )
                                print(
                                    f"Attempted formats: {', '.join(date_formats)}",
                                    file=sys.stderr,
                                )
                                print(
                                    f"Errors: {'; '.join(error_messages)}",
                                    file=sys.stderr,
                                )
                        else:
                            print(
                                f"Warning: Date column '{date_col}' is empty in {tbl}",
                                file=sys.stderr,
                            )
                    else:
                        # Try to find a case-insensitive match
                        columns_lower = [col.lower() for col in sample_df.columns]
                        if date_col.lower() in columns_lower:
                            # Found case-insensitive match
                            actual_col = sample_df.columns[
                                columns_lower.index(date_col.lower())
                            ]
                            print(
                                f"Warning: Column name case mismatch in {tbl}. Expected '{date_col}', found '{actual_col}'.",
                                file=sys.stderr,
                            )
                            print(
                                f"Please update your configuration to use the correct case: '{actual_col}'",
                                file=sys.stderr,
                            )
                            date_col = actual_col

                            # Continue with the correct column name
                            # Try each format until one works
                            if not sample_df[date_col].is_empty():
                                for fmt in date_formats:
                                    try:
                                        test_df = sample_df.with_columns(
                                            [
                                                pl.col(date_col)
                                                .str.strptime(pl.Datetime, fmt)
                                                .alias(f"{date_col}_parsed")
                                            ]
                                        )
                                        success = True
                                        fmt_to_use = fmt
                                        print(
                                            f"\u2713 Using date format '{fmt}' for column '{date_col}'"
                                        )
                                        break
                                    except Exception as e:
                                        error_messages.append(
                                            f"Format '{fmt}': {str(e)}"
                                        )
                            else:
                                print(
                                    f"Warning: Date column '{date_col}' is empty in {tbl}",
                                    file=sys.stderr,
                                )
                        else:
                            print(
                                f"Warning: Date column '{date_col}' not found in {tbl}",
                                file=sys.stderr,
                            )
                            print(
                                f"Available columns: {', '.join(sample_df.columns)}",
                                file=sys.stderr,
                            )
                except Exception as e:
                    print(
                        f"Warning: Error analyzing sample data for {tbl}: {str(e)}",
                        file=sys.stderr,
                    )

            # Partition by year if date_col available and successfully parsed
            writer_opts = {}
            if date_col and success:
                writer_opts = {"partition_by": [date_col]}

            if use_chunking:
                # Process in chunks for large files
                import time

                start_time = time.time()

                print(f"Processing {tbl} in chunks of {CHUNK_SIZE} rows...")
                print(f"File size: {file_size / (1024*1024):.2f} MB")

                # Process and write the first chunk
                chunk_start = time.time()
                first_chunk_scan = pl.scan_csv(
                    src_file,
                    infer_schema_length=10000,
                    ignore_errors=True,
                    n_rows=CHUNK_SIZE,
                )

                # Apply date parsing if needed
                if date_col and success and fmt_to_use:
                    first_chunk_scan = first_chunk_scan.with_columns(
                        [
                            pl.col(date_col)
                            .str.strptime(pl.Datetime, fmt_to_use)
                            .alias(date_col)
                        ]
                    )

                # Collect and write the first chunk with write_mode="overwrite"
                first_chunk = first_chunk_scan.collect()
                try:
                    first_chunk.write_parquet(
                        output_path, compression="zstd", **writer_opts
                    )
                    chunk_time = time.time() - chunk_start
                    print(
                        f"\u2713 Wrote first chunk ({len(first_chunk)} rows) in {chunk_time:.2f} seconds"
                    )
                except Exception as e:
                    if (
                        "already exists" in str(e).lower()
                        or "not a directory" in str(e).lower()
                        or "os error 20" in str(e).lower()
                    ):
                        # Clean up existing file/directory
                        import shutil

                        if output_path.exists():
                            print(f"Cleaning up existing path {output_path}")
                            if output_path.is_dir():
                                shutil.rmtree(output_path)
                            else:
                                output_path.unlink()
                        # Make sure parent directory exists
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        # Try writing again
                        first_chunk.write_parquet(
                            output_path, compression="zstd", **writer_opts
                        )
                        chunk_time = time.time() - chunk_start
                        print(
                            f"\u2713 Cleaned up path and wrote first chunk ({len(first_chunk)} rows) in {chunk_time:.2f} seconds"
                        )
                    else:
                        # Re-raise other exceptions
                        raise

                # Process remaining chunks
                total_rows = len(first_chunk)
                offset = CHUNK_SIZE
                chunk_num = 1

                # Estimate total chunks and rows
                # Very rough estimate based on the first chunk's size
                estimated_total_rows = (
                    int(file_size / (offset / len(first_chunk)))
                    if len(first_chunk) > 0
                    else 0
                )
                estimated_chunks = estimated_total_rows // CHUNK_SIZE + 1
                print(
                    f"Estimated total: ~{estimated_total_rows:,} rows in ~{estimated_chunks} chunks"
                )
                print(f"Progress: [{'·' * 20}] 0%")

                # Define maximum retry attempts for failed chunks
                MAX_RETRIES = 3

                while True:
                    chunk_start = time.time()
                    retry_count = 0
                    success = False

                    # Try processing this chunk with retries
                    while retry_count < MAX_RETRIES and not success:
                        try:
                            # Reduce chunk size on retries
                            current_chunk_size = CHUNK_SIZE
                            if retry_count > 0:
                                # Reduce by half on each retry
                                current_chunk_size = max(
                                    1000, CHUNK_SIZE // (2**retry_count)
                                )
                                print(
                                    f"\nRetry {retry_count}/{MAX_RETRIES}: Reduced chunk size to {current_chunk_size} rows"
                                )

                            chunk_scan = pl.scan_csv(
                                src_file,
                                infer_schema_length=10000,
                                ignore_errors=True,
                                n_rows=current_chunk_size,
                                skip_rows=offset,
                            )

                            # Apply date parsing if needed
                            if date_col and fmt_to_use:
                                chunk_scan = chunk_scan.with_columns(
                                    [
                                        pl.col(date_col)
                                        .str.strptime(pl.Datetime, fmt_to_use)
                                        .alias(date_col)
                                    ]
                                )

                            # Collect the chunk
                            chunk = chunk_scan.collect()

                            # Break if the chunk is empty
                            if len(chunk) == 0:
                                success = True
                                break

                            # Append to the Parquet file
                            try:
                                chunk.write_parquet(
                                    output_path,
                                    compression="zstd",
                                    mode="append",
                                    **writer_opts,
                                )
                                chunk_time = time.time() - chunk_start
                                success = True
                            except Exception as append_error:
                                if (
                                    "already exists" in str(append_error).lower()
                                    or "not a directory" in str(append_error).lower()
                                    or "os error 20" in str(append_error).lower()
                                ):
                                    print(
                                        f"\n\u2717 Error appending to parquet file: {str(append_error)}",
                                        file=sys.stderr,
                                    )
                                    print("Attempting to fix directory structure...")

                                    # Try to fix partitioned directories
                                    import glob
                                    import shutil

                                    # Look for problematic partitioned directories
                                    partition_dirs = [
                                        p
                                        for p in glob.glob(f"{output_path}/*/")
                                        if not Path(p).is_dir()
                                    ]
                                    for problem_path in partition_dirs:
                                        print(
                                            f"Removing problematic path: {problem_path}"
                                        )
                                        if Path(problem_path).exists():
                                            if Path(problem_path).is_file():
                                                Path(problem_path).unlink()
                                            else:
                                                shutil.rmtree(
                                                    problem_path, ignore_errors=True
                                                )

                                    # Try writing again
                                    chunk.write_parquet(
                                        output_path,
                                        compression="zstd",
                                        mode="append",
                                        **writer_opts,
                                    )
                                    chunk_time = time.time() - chunk_start
                                    success = True
                                    print(
                                        "\u2713 Fixed directory structure and appended chunk successfully"
                                    )
                                else:
                                    # Re-raise other errors
                                    raise append_error
                        except Exception as e:
                            retry_count += 1
                            if retry_count >= MAX_RETRIES:
                                print(
                                    f"\n\u2717 Failed to process chunk after {MAX_RETRIES} attempts: {str(e)}",
                                    file=sys.stderr,
                                )
                                # Try to continue with the next chunk
                                offset += current_chunk_size
                                print(f"Skipping to next chunk at offset {offset}")
                                break
                            else:
                                print(
                                    f"\n\u2717 Error processing chunk: {str(e)}. Retrying...",
                                    file=sys.stderr,
                                )

                    # If we couldn't process this chunk after retries, continue to the next one
                    if not success and retry_count >= MAX_RETRIES:
                        continue

                    # If the chunk was empty, we're done
                    if len(chunk) == 0:
                        break

                    # Update counters
                    total_rows += len(chunk)
                    offset += CHUNK_SIZE
                    chunk_num += 1

                    # Calculate and print progress
                    progress_pct = (
                        min(100, (chunk_num / estimated_chunks) * 100)
                        if estimated_chunks > 0
                        else 0
                    )
                    progress_bar_length = 20
                    filled_length = int(progress_pct / 100 * progress_bar_length)
                    bar = "█" * filled_length + "·" * (
                        progress_bar_length - filled_length
                    )

                    # Estimate time remaining
                    elapsed_time = time.time() - start_time
                    rows_per_sec = total_rows / elapsed_time if elapsed_time > 0 else 0
                    est_total_time = (
                        estimated_total_rows / rows_per_sec if rows_per_sec > 0 else 0
                    )
                    est_remaining = max(0, est_total_time - elapsed_time)

                    # Format time as h:m:s
                    def format_time(seconds):
                        h = int(seconds // 3600)
                        m = int((seconds % 3600) // 60)
                        s = int(seconds % 60)
                        return f"{h:02d}:{m:02d}:{s:02d}"

                    # Clear previous line and print progress
                    print(
                        f"\rChunk {chunk_num}/{estimated_chunks} | Progress: [{bar}] {progress_pct:.1f}% | "
                        f"Rows: {total_rows:,}/{estimated_total_rows:,} | "
                        f"Speed: {rows_per_sec:.1f} rows/s | "
                        f"Elapsed: {format_time(elapsed_time)} | "
                        f"Est. remaining: {format_time(est_remaining)}",
                        end="",
                    )

                # Final stats
                total_time = time.time() - start_time
                print(
                    f"\n\n\u2713 Completed processing {total_rows:,} rows in {format_time(total_time)}"
                )
                print(f"\u2713 Average speed: {total_rows/total_time:.1f} rows/second")
                print(f"\u2713 Output saved to {output_path}")
                print()
            else:
                # Regular processing for smaller files
                scan = pl.scan_csv(
                    src_file, infer_schema_length=10000, ignore_errors=True
                )

                # Apply date parsing if needed
                if date_col and success and fmt_to_use:
                    scan = scan.with_columns(
                        [
                            pl.col(date_col)
                            .str.strptime(pl.Datetime, fmt_to_use)
                            .alias(date_col)
                        ]
                    )

                # Collect and write in one go
                try:
                    scan.collect().write_parquet(
                        output_path, compression="zstd", **writer_opts
                    )
                except Exception as e:
                    if (
                        "already exists" in str(e).lower()
                        or "not a directory" in str(e).lower()
                        or "os error 20" in str(e).lower()
                    ):
                        # Clean up existing file/directory
                        import shutil

                        if output_path.exists():
                            print(f"Cleaning up existing path {output_path}")
                            if output_path.is_dir():
                                shutil.rmtree(output_path)
                            else:
                                output_path.unlink()
                        # Make sure parent directory exists
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        # Try writing again
                        scan.collect().write_parquet(
                            output_path, compression="zstd", **writer_opts
                        )
                    else:
                        # Re-raise other exceptions
                        raise

            outputs[tbl] = output_path
            print(f"\u2713 Successfully converted {tbl}")

        except Exception as e:
            print(f"\u2717 Error converting {tbl}: {str(e)}", file=sys.stderr)

    return outputs


def build_bigjoin(parquet_files: dict, output_dir: Path) -> Path:
    """
    Build the kernel-competition-dataset bigjoin table.

    Args:
        parquet_files: Dictionary of input Parquet file paths
        output_dir: Output directory for the bigjoin

    Returns:
        Path: Path to the output bigjoin Parquet file
    """
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
    ]  # Note: 'Datasets' table no longer required
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

    # Build the bigjoin
    sql = f"""
    CREATE OR REPLACE TABLE kernel_bigjoin AS
    SELECT
        kv.Id               AS kernel_version_id,
        kv.CreationDate     AS kernel_ts,
        kv.AuthorUserId     AS author_id,
        k.TotalVotes        AS total_votes,
        k.TotalViews        AS total_views,
        c.Id                AS comp_id,
        c.Title             AS comp_title,
        c.endDate       AS comp_enabled_ts,
        kv.isCommit         AS is_commit,
        kvds.SourceDatasetVersionId AS dataset_id
    FROM read_parquet('{_duckdb_read_path(parquet_files['KernelVersions'])}') AS kv
    LEFT JOIN read_parquet('{_duckdb_read_path(parquet_files['Kernels'])}') AS k ON k.CurrentKernelVersionId = kv.Id
    LEFT JOIN read_parquet('{_duckdb_read_path(parquet_files['KernelVersionCompetitionSources'])}') AS kvcs ON kvcs.KernelVersionId = kv.Id
    LEFT JOIN read_parquet('{_duckdb_read_path(parquet_files['Competitions'])}') AS c ON c.Id = kvcs.sourceCompetitionId
    LEFT JOIN read_parquet('{_duckdb_read_path(parquet_files['KernelVersionDatasetSources'])}') AS kvds ON kvds.KernelVersionId = kv.Id
    """
    try:
        # Execute the main query
        con.sql(sql)
        
        # Write the result to a Parquet file
        con.sql(f"COPY kernel_bigjoin TO '{output_path}' (FORMAT 'parquet')")

    except Exception as e:
        print(f"✗ Failed to execute SQL query: {str(e)}", file=sys.stderr)
        # Add more detailed diagnosis
        if "Table with name" in str(e) and "does not exist" in str(e):
            missing_table = str(e).split("Table with name ")[1].split(" does not exist!")[0]
            print("\nDiagnosis: Missing table '{missing_table}'")
            print("This could be due to:")
            print("1. The table wasn't registered correctly")
            print("2. The table name case is incorrect in the SQL query")
            print("3. The CSV file wasn't processed in the csv_to_parquet step")
        raise

    # Return the path to the bigjoin file
    return output_path


def prune_columns(input_path: Path, output_path: Path) -> Path:
    """
    Prune columns and fix data types.

    Args:
        input_path: Path to the input bigjoin Parquet file
        output_path: Path to output the cleaned file

    Returns:
        Path: Path to the output cleaned Parquet file
    """
    # Read table
    tbl = pq.read_table(input_path)

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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pl.from_arrow(tbl).write_parquet(output_path, compression="zstd")

    return output_path


def create_mini_meta(
    input_path: Path, output_path: Path, sample_frac: float = 0.01
) -> Path:
    """
    Create a mini-Meta sample dataset.

    Args:
        input_path: Path to input Parquet file
        output_path: Path to output sample file
        sample_frac: Fraction of data to sample (default: 1%)

    Returns:
        Path: Path to the output sample Parquet file
    """
    # Create destination directory if it doesn't exist
    output_path.parent.mkdir(exist_ok=True, parents=True)

    # Use hash-based sampling for consistent results
    scan = pl.scan_parquet(input_path)
    sampled = (
        scan.with_columns(
            (pl.col("kernel_version_id").hash().abs() % 100 < int(sample_frac * 100)).alias(
                "_sample"
            )
        )
        .filter(pl.col("_sample") == True)
        .drop("_sample")
        .collect()
    )

    # Write to parquet
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


def validate_data(input_path: Path) -> bool:
    """
    Run basic validation checks on the cleaned data.

    Args:
        input_path: Path to the cleaned bigjoin Parquet file

    Returns:
        bool: True if all checks pass, False otherwise
    """
    # Lightweight validation – ensure the Parquet file can be read and is non-empty
    try:
        df = pl.read_parquet(input_path, n_rows=1)
        return df.height > 0
    except Exception:
        return False
