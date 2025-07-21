#!/usr/bin/env python
"""
Script to run the Kedro pipeline for the Meta Kaggle project.
"""
import logging
import os
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add the project root to the path so we can import from src
project_root = Path(__file__).parent.parent
os.chdir(project_root)  # Change to project root directory
sys.path.append(str(project_root))

try:
    from kedro.framework.session import KedroSession
    from kedro.framework.startup import bootstrap_project
except ImportError as e:
    logger.error(f"Could not import Kedro packages: {e}")
    logger.error(
        "Make sure you've installed kedro and activated your conda environment."
    )
    sys.exit(1)


def main():
    """Run the Kedro pipeline."""
    # Check if data directories exist
    data_dirs = [
        "data/raw_csv",
        "data/parquet/raw",
        "data/intermediate",
        "data/mini_meta",
    ]

    for d in data_dirs:
        path = project_root / d
        if not path.exists():
            logger.warning(f"Directory {path} does not exist. Creating...")
            path.mkdir(parents=True, exist_ok=True)

    # Check if raw_csv has files
    raw_csv_path = project_root / "data/raw_csv"
    csv_files = list(raw_csv_path.glob("*.csv"))
    if not csv_files:
        logger.warning("No CSV files found in data/raw_csv.")
        logger.warning(
            "Please make sure to place the Meta Kaggle CSV files there before running the pipeline."
        )
        return

    # Bootstrap the project
    metadata = bootstrap_project(project_root)

    # Start a KedroSession with custom config paths
    logger.info("Starting Kedro session...")
    conf_path = project_root / "conf"
    with KedroSession.create(
        metadata.package_name, project_root, env="base", conf_source=conf_path
    ) as session:
        logger.info("Running data_layer pipeline...")
        session.run(pipeline_name="data_layer")
        logger.info("Data layer pipeline completed successfully!")


if __name__ == "__main__":
    main()
