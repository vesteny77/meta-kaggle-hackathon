"""
Tests for the gcs_path_utils module.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from src.features.gcs_path_utils import GCSFileHandler, read_kernel_code


@pytest.fixture
def mock_gcs_client():
    """Create a mock GCS client."""
    with patch('src.features.gcs_path_utils.storage.Client.create_anonymous_client') as mock_client:
        mock_bucket = MagicMock()
        mock_client.return_value.bucket.return_value = mock_bucket
        yield mock_bucket


def test_gcs_file_handler_init(mock_gcs_client):
    """Test GCSFileHandler initialization."""
    handler = GCSFileHandler("test-bucket")
    assert handler.bucket == mock_gcs_client


def test_list_paths(mock_gcs_client):
    """Test listing paths in the GCS bucket."""
    # Configure mock
    mock_blob1 = Mock()
    mock_blob1.name = "path/to/file1.py"
    mock_blob2 = Mock()
    mock_blob2.name = "path/to/file2.py"
    
    mock_gcs_client.list_blobs.return_value = [mock_blob1, mock_blob2]
    
    # Test
    handler = GCSFileHandler("test-bucket")
    paths = handler.list_paths(prefix="path/to", max_results=5)
    
    # Verify
    assert paths == ["path/to/file1.py", "path/to/file2.py"]
    mock_gcs_client.list_blobs.assert_called_once_with(prefix="path/to", max_results=5)


def test_read_file(mock_gcs_client):
    """Test reading a file from GCS."""
    # Configure mock
    mock_blob = Mock()
    mock_blob.download_as_text.return_value = "file content"
    mock_gcs_client.blob.return_value = mock_blob
    
    # Test
    handler = GCSFileHandler("test-bucket")
    content = handler.read_file("path/to/file.py")
    
    # Verify
    assert content == "file content"
    mock_gcs_client.blob.assert_called_once_with("path/to/file.py")
    mock_blob.download_as_text.assert_called_once_with(encoding="utf-8")


def test_read_file_fallback(mock_gcs_client):
    """Test reading a file with fallback to binary."""
    # Configure mock
    mock_blob = Mock()
    mock_blob.download_as_text.side_effect = Exception("Text read failed")
    mock_blob.download_as_bytes.return_value = b"binary content"
    mock_gcs_client.blob.return_value = mock_blob
    
    # Test
    handler = GCSFileHandler("test-bucket")
    content = handler.read_file("path/to/file.bin")
    
    # Verify
    assert content == "binary content"
    mock_blob.download_as_bytes.assert_called_once()


def test_read_file_error(mock_gcs_client):
    """Test reading a file with both methods failing."""
    # Configure mock
    mock_blob = Mock()
    mock_blob.download_as_text.side_effect = Exception("Text read failed")
    mock_blob.download_as_bytes.side_effect = Exception("Binary read failed")
    mock_gcs_client.blob.return_value = mock_blob
    
    # Test
    handler = GCSFileHandler("test-bucket")
    with pytest.raises(IOError):
        handler.read_file("path/to/nonexistent.file")


def test_file_exists(mock_gcs_client):
    """Test checking if a file exists in GCS."""
    # Configure mock
    mock_blob = Mock()
    mock_blob.exists.return_value = True
    mock_gcs_client.blob.return_value = mock_blob
    
    # Test
    handler = GCSFileHandler("test-bucket")
    exists = handler.file_exists("path/to/file.py")
    
    # Verify
    assert exists is True
    mock_gcs_client.blob.assert_called_once_with("path/to/file.py")
    mock_blob.exists.assert_called_once()


def test_get_kernel_path_py(mock_gcs_client):
    """Test finding a Python kernel path."""
    # Configure mock
    mock_blob = Mock()
    mock_blob.exists.side_effect = [True, False, False, False]  # .py exists, others don't
    mock_gcs_client.blob.return_value = mock_blob
    
    # Test
    handler = GCSFileHandler("test-bucket")
    path = handler.get_kernel_path(12345)
    
    # Verify
    assert path == "0001/234/12345.py"


def test_get_kernel_path_ipynb(mock_gcs_client):
    """Test finding a Jupyter notebook kernel path."""
    # Configure mock
    mock_blob = Mock()
    mock_blob.exists.side_effect = [False, True, False, False]  # .ipynb exists, others don't
    mock_gcs_client.blob.return_value = mock_blob
    
    # Test
    handler = GCSFileHandler("test-bucket")
    path = handler.get_kernel_path(12345)
    
    # Verify
    assert path == "0001/234/12345.ipynb"


def test_get_kernel_path_r(mock_gcs_client):
    """Test finding an R kernel path."""
    # Configure mock
    mock_blob = Mock()
    mock_blob.exists.side_effect = [False, False, True, False]  # .r exists, others don't
    mock_gcs_client.blob.return_value = mock_blob
    
    # Test
    handler = GCSFileHandler("test-bucket")
    path = handler.get_kernel_path(12345)
    
    # Verify
    assert path == "0001/234/12345.r"


def test_get_kernel_path_rmd(mock_gcs_client):
    """Test finding an Rmd kernel path."""
    # Configure mock
    mock_blob = Mock()
    mock_blob.exists.side_effect = [False, False, False, True]  # .rmd exists, others don't
    mock_gcs_client.blob.return_value = mock_blob
    
    # Test
    handler = GCSFileHandler("test-bucket")
    path = handler.get_kernel_path(12345)
    
    # Verify
    assert path == "0001/234/12345.rmd"


def test_get_kernel_path_none(mock_gcs_client):
    """Test when no kernel path is found."""
    # Configure mock
    mock_blob = Mock()
    mock_blob.exists.return_value = False
    mock_gcs_client.blob.return_value = mock_blob
    
    # Test
    handler = GCSFileHandler("test-bucket")
    path = handler.get_kernel_path(12345)
    
    # Verify
    assert path is None


@pytest.fixture
def mock_gcs_handler():
    """Create a mock GCSFileHandler."""
    handler = Mock()
    return handler


def test_read_kernel_code_python(mock_gcs_handler):
    """Test reading Python code."""
    # Configure mock
    mock_gcs_handler.get_kernel_path.return_value = "0001/234/12345.py"
    mock_gcs_handler.read_file.return_value = "import numpy as np\n\nprint('Hello')"
    
    # Test
    code, md_cells, ext = read_kernel_code(mock_gcs_handler, 12345)
    
    # Verify
    assert code == "import numpy as np\n\nprint('Hello')"
    assert md_cells == []
    assert ext == ".py"


def test_read_kernel_code_ipynb(mock_gcs_handler):
    """Test reading Jupyter notebook code."""
    # Create mock notebook content
    notebook_content = {
        "cells": [
            {"cell_type": "markdown", "source": ["# Title", "Description"]},
            {"cell_type": "code", "source": ["import pandas as pd", "df = pd.DataFrame()"]},
            {"cell_type": "markdown", "source": ["## Section"]},
            {"cell_type": "code", "source": ["print('Hello')"]},
        ]
    }
    
    # Configure mock
    mock_gcs_handler.get_kernel_path.return_value = "0001/234/12345.ipynb"
    mock_gcs_handler.read_file.return_value = json.dumps(notebook_content)
    
    # Test
    code, md_cells, ext = read_kernel_code(mock_gcs_handler, 12345)
    
    # Verify
    assert code == "import pandas as pd\ndf = pd.DataFrame()\nprint('Hello')"
    assert md_cells == ["# Title\nDescription", "## Section"]
    assert ext == ".ipynb"


def test_read_kernel_code_nonexistent(mock_gcs_handler):
    """Test reading nonexistent kernel code."""
    # Configure mock
    mock_gcs_handler.get_kernel_path.return_value = None
    
    # Test
    code, md_cells, ext = read_kernel_code(mock_gcs_handler, 12345)
    
    # Verify
    assert code is None
    assert md_cells == []
    assert ext is None


def test_read_kernel_code_error(mock_gcs_handler):
    """Test error handling when reading kernel code."""
    # Configure mock
    mock_gcs_handler.get_kernel_path.return_value = "0001/234/12345.ipynb"
    mock_gcs_handler.read_file.side_effect = Exception("Error reading file")
    
    # Test
    code, md_cells, ext = read_kernel_code(mock_gcs_handler, 12345)
    
    # Verify
    assert code is None
    assert md_cells == []
    assert ext is None


def test_read_kernel_code_ipynb_malformed(mock_gcs_handler):
    """Test reading malformed Jupyter notebook."""
    # Configure mock
    mock_gcs_handler.get_kernel_path.return_value = "0001/234/12345.ipynb"
    mock_gcs_handler.read_file.return_value = "{invalid json"
    
    # Test
    code, md_cells, ext = read_kernel_code(mock_gcs_handler, 12345)
    
    # Verify
    assert code is None
    assert md_cells == []
    assert ext == ".ipynb"