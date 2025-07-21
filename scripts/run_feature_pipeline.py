#!/usr/bin/env python
"""Convenience launcher for the **feature extraction** Kedro pipeline.

Run from project root:

    python scripts/run_feature_pipeline.py [--local-code-root data/raw_code]

It ensures the project is on `PYTHONPATH`, creates missing data folders, and
invokes `kedro run --pipeline feature_extraction` programmatically via Kedro's
Python API (so we don't require the CLI to be installed in the environment).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from kedro.framework.session import KedroSession
    from kedro.framework.startup import bootstrap_project
except ImportError as e:
    logger.error("Kedro is not installed: %s", e)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run feature extraction Kedro pipeline")
    parser.add_argument("--local-code-root", default="data/raw_code", help="Path to kernel code directory")
    args = parser.parse_args()

    # Ensure local code root exists
    lroot = PROJECT_ROOT / args.local_code_root
    if not lroot.exists():
        logger.error("Local code directory %s does not exist.", lroot)
        sys.exit(1)

    # Kedro config variable so nodes can pick it up via params
    os.environ["KEDRO_CONFIG_PATTERNS"] = "**/conf/**/*"

    # Bootstrap project & start session
    metadata = bootstrap_project(PROJECT_ROOT)
    conf_path = PROJECT_ROOT / "conf"
    with KedroSession.create(metadata.package_name, PROJECT_ROOT, env="base", conf_source=conf_path) as session:
        logger.info("Running feature_extraction pipeline …")
        session.run(pipeline_name="features")
    logger.info("Feature extraction pipeline finished.")


if __name__ == "__main__":
    main() 