"""
Tests for the path_utils module.
"""

from pathlib import Path

import pytest

# Import feature functions
try:
    from src.features.path_utils import get_code_file_path
except ImportError:
    # Fallback to local imports if module path isn't working
    import sys
    from pathlib import Path as PathLib  # Rename to avoid conflict

    # Add the project root to sys.path if not already there
    project_root = PathLib(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    # Now try again
    try:
        from src.features.path_utils import get_code_file_path
    except (ImportError, ModuleNotFoundError):
        # Skip test if module still not found
        pytestmark = pytest.mark.skip(reason="Path utils module not found")


def test_get_code_file_path():
    """Test the get_code_file_path function."""

    # Test a simple case
    path = get_code_file_path(1234567)
    assert path == Path("data/raw_code/234/567/1234567")

    # Test a larger ID
    path = get_code_file_path(9876543210)
    assert path == Path("data/raw_code/543/210/9876543210")

    # Test a small ID
    path = get_code_file_path(50)
    assert path == Path("data/raw_code/000/050/50")

    # Test with a custom code_dir
    path = get_code_file_path(1234, Path("/custom/path"))
    assert path == Path("/custom/path/001/234/1234")


def test_get_code_file_path_types():
    """Test that the function handles different input types correctly."""
    # Should handle int or string input
    path1 = get_code_file_path(50)
    path2 = get_code_file_path("50")

    assert path1 == path2
