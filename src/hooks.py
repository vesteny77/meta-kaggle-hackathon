"""Project hooks."""

from typing import Any, Dict

from kedro.framework.hooks import hook_impl


class ProjectHooks:
    """Namespace for project hooks."""

    @hook_impl
    def register_pipelines(self) -> Dict[str, Any]:
        """Register the project's pipelines.

        Returns:
            A mapping from pipeline names to ``Pipeline`` objects.
        """
        from src.pipeline_registry import register_pipelines

        return register_pipelines()


hooks = ProjectHooks()
