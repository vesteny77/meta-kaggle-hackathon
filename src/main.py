"""
Main entry point for running kedro pipelines.
"""

import logging
from pathlib import Path

from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project


def run_pipeline(pipeline_name=None, tags=None, env=None):
    """
    Run the specified pipeline.

    Args:
        pipeline_name: Name of the pipeline to run
        tags: Tags to filter nodes
        env: Kedro environment to use
    """
    project_path = Path.cwd()
    metadata = bootstrap_project(project_path)
    configure_project(metadata.package_name)

    with KedroSession.create(metadata.package_name, project_path, env=env) as session:
        logging.info(f"Running pipeline: {pipeline_name}")
        session.run(pipeline_name=pipeline_name, tags=tags)


if __name__ == "__main__":
    import sys

    pipeline_name = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(pipeline_name)
