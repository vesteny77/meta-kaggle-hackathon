#!/usr/bin/env python
"""
Run the graph analytics pipeline for the Meta Kaggle project.

This script executes the graph analytics pipeline, which builds graph representations
of the Kaggle ecosystem and analyzes relationships like forks, competition-kernel
connections, team collaborations, dataset gravity, and forum-code knowledge transfer.

Example:
    $ python run_graphs_pipeline.py
    $ python run_graphs_pipeline.py --node-limit 1000
    $ python run_graphs_pipeline.py --libraries gbdt dl_pytorch
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_pipeline(node_limit=None, libraries=None, time_decay_hours=None):
    """
    Run the graph analytics pipeline with optional parameters.
    
    Args:
        node_limit: Maximum number of nodes to include in graph visualizations
        libraries: List of libraries to analyze for diffusion patterns
        time_decay_hours: Time decay parameter for edge weights (in hours)
    """
    logger.info("Bootstrapping Kedro project")
    metadata = bootstrap_project(project_root)
    
    # Create parameter overrides if needed
    params = {}
    if node_limit is not None or libraries is not None or time_decay_hours is not None:
        params["graphs"] = {}
        
        if node_limit is not None:
            params["graphs"]["node_limit"] = node_limit
            
        if libraries is not None:
            params["graphs"]["libraries_to_analyze"] = libraries
            
        if time_decay_hours is not None:
            params["graphs"]["time_decay_hours"] = time_decay_hours
    
    # Run pipeline
    with KedroSession.create(metadata.package_name, project_path=project_root) as session:
        logger.info("Running graphs pipeline")
        session.run(pipeline_name="graphs", params=params)
    
    logger.info("Graph analytics pipeline completed successfully")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the graph analytics pipeline for Meta Kaggle project"
    )
    
    parser.add_argument(
        "--node-limit", 
        type=int, 
        help="Maximum number of nodes to include in graph visualizations"
    )
    
    parser.add_argument(
        "--libraries", 
        nargs="+", 
        help="Libraries to analyze for diffusion patterns (e.g., gbdt dl_pytorch)"
    )
    
    parser.add_argument(
        "--time-decay", 
        type=int, 
        dest="time_decay_hours",
        help="Time decay parameter for edge weights (in hours)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        node_limit=args.node_limit,
        libraries=args.libraries,
        time_decay_hours=args.time_decay_hours
    )