"""
Tests for the path_utils module.
"""
import pytest
from pathlib import Path
from src.features.path_utils import get_code_file_path


def test_get_code_file_path():
    """Test that the get_code_file_path function returns the correct path."""
    # Test a simple case
    path = get_code_file_path(1234567)
    assert path == Path("data/raw_code/0001/234/1234567")
    
    # Test a larger ID
    path = get_code_file_path(9876543210)
    assert path == Path("data/raw_code/9876/543/9876543210")
    
    # Test a small ID (should pad with zeros)
    path = get_code_file_path(50)
    assert path == Path("data/raw_code/0000/050/50")
    
    # Test with a custom code_dir
    path = get_code_file_path(1234, Path("/custom/path"))
    assert path == Path("/custom/path/0000/001/1234")


def test_get_code_file_path_types():
    """Test that the function handles different input types correctly."""
    # Should handle int or string input
    path1 = get_code_file_path(50)
    path2 = get_code_file_path("50")
    
    assert path1 == path2