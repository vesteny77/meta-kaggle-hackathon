#!/usr/bin/env python
"""Fast downloader for the public Meta-Kaggle code dataset via **kagglehub**.

This script relies on the kaggle-maintained `kagglehub` package which provides
`dataset_download()` - it automatically:
 • Finds the latest dataset version
 • Streams each shard in parallel
 • Extracts the archives under a local cache directory

We simply set the cache directory to ``data/raw_code`` and launch the download.

Usage (from repo root):
    python scripts/download_extract_gcs_archives.py

You can override the cache directory with ``--cache-dir`` if desired.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import kagglehub
from tqdm import tqdm

DEFAULT_CACHE = Path("data/raw_code")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Meta-Kaggle code dataset using kagglehub")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE), help="Local cache location (KAGGLEHUB_CACHE)")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Tell kagglehub where to cache files
    os.environ["KAGGLEHUB_CACHE"] = str(cache_dir)

    print("Starting download …\nCache dir:", cache_dir)
    # dataset_download already prints progress, but we wrap with tqdm for clarity
    try:
        path = kagglehub.dataset_download("kaggle/meta-kaggle-code")
    except Exception as exc:
        print("[ERROR] dataset_download failed:", exc)
        raise

    print("\nDownload + extraction complete.")
    print("Path to dataset files:", path)


if __name__ == "__main__":
    main() 