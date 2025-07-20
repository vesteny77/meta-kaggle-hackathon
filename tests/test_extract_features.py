"""
Tests for the extract_features module.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import polars as pl

from src.features.extract_features import (
    detect_imports, 
    tag_libraries, 
    tag_architectures, 
    complexity_metrics, 
    embed_markdown,
    read_code,
    GCSFileHandler,
    process_kernel_batch
)


def test_detect_imports_ast():
    """Test import detection using AST parser."""
    code = """
    import numpy as np
    import pandas
    from sklearn import metrics
    from tensorflow.keras import layers
    """
    imports = detect_imports(code)
    assert "numpy" in imports
    assert "pandas" in imports
    assert "sklearn" in imports
    assert "tensorflow" in imports


def test_detect_imports_regex_fallback():
    """Test import detection using regex fallback."""
    # This code has a syntax error that will trigger the regex fallback
    code = """
    import numpy as np
    from pandas import DataFrame
    if True:
        print("hello"
    import matplotlib.pyplot as plt
    """
    imports = detect_imports(code)
    assert "numpy" in imports
    assert "pandas" in imports
    assert "matplotlib" not in imports  # After syntax error, regex won't catch this


def test_detect_imports_empty():
    """Test import detection with empty code."""
    imports = detect_imports("")
    assert imports == set()
    
    imports = detect_imports(None)
    assert imports == set()


def test_tag_libraries():
    """Test library tagging."""
    imports = {"numpy", "pandas", "torch", "transformers", "cv2"}
    tags = tag_libraries(imports)
    
    assert tags["dl_pytorch"] is True
    assert tags["nlp"] is True
    assert tags["vision"] is True
    assert tags["data_proc"] is True
    assert tags["gbdt"] is False  # Not in imports


def test_tag_libraries_empty():
    """Test library tagging with empty imports."""
    imports = set()
    tags = tag_libraries(imports)
    
    for tag_value in tags.values():
        assert tag_value is False


def test_tag_architectures():
    """Test architecture pattern detection."""
    code = """
    model = ResNet50(weights='imagenet')
    attention = MultiHeadAttention(8, 64)
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    """
    
    tags = tag_architectures(code)
    
    assert tags["ResNet"] is True
    assert tags["Attention"] is True
    assert tags["BERT"] is True
    assert tags["UNet"] is False  # Not in code


def test_tag_architectures_empty():
    """Test architecture detection with empty code."""
    tags = tag_architectures("")
    
    for tag_value in tags.values():
        assert tag_value is False
    
    tags = tag_architectures(None)
    for tag_value in tags.values():
        assert tag_value is False


def test_complexity_metrics():
    """Test code complexity metrics calculation."""
    code = """
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
"""
    metrics = complexity_metrics(code)
    
    assert metrics["loc"] > 0
    assert metrics["mean_cc"] > 0
    assert metrics["max_cc"] > 0


def test_complexity_metrics_empty():
    """Test complexity metrics with empty code."""
    metrics = complexity_metrics("")
    
    assert metrics["loc"] == 0
    assert metrics["mean_cc"] == 0
    assert metrics["max_cc"] == 0
    
    metrics = complexity_metrics(None)
    assert metrics["loc"] == 0
    assert metrics["mean_cc"] == 0
    assert metrics["max_cc"] == 0


@pytest.fixture
def mock_sentence_transformer():
    """Create a mock SentenceTransformer."""
    with patch('src.features.extract_features.SentenceTransformer') as mock_st:
        model = Mock()
        model.encode.return_value = [0.1, 0.2, 0.3, 0.4]
        mock_st.return_value = model
        yield model


def test_embed_markdown(mock_sentence_transformer):
    """Test markdown embedding."""
    md_list = ["# Title", "This is a test", "## Section"]
    
    embedding = embed_markdown(md_list, mock_sentence_transformer)
    
    assert embedding == [0.1, 0.2, 0.3, 0.4]
    mock_sentence_transformer.encode.assert_called_once()


def test_embed_markdown_empty():
    """Test markdown embedding with empty list."""
    model = Mock()
    embedding = embed_markdown([], model)
    
    assert embedding is None
    model.encode.assert_not_called()


def test_embed_markdown_error(mock_sentence_transformer):
    """Test error handling in markdown embedding."""
    mock_sentence_transformer.encode.side_effect = Exception("Embedding error")
    
    embedding = embed_markdown(["# Error test"], mock_sentence_transformer)
    
    assert embedding is None


@pytest.fixture
def mock_gcs_handler():
    """Create a mock GCSFileHandler."""
    with patch('src.features.extract_features.GCSFileHandler') as mock_handler_class:
        handler = Mock()
        mock_handler_class.return_value = handler
        yield handler


def test_read_code_python(mock_gcs_handler):
    """Test reading Python code."""
    # Configure mock
    mock_gcs_handler.get_kernel_path.return_value = "0001/234/12345.py"
    mock_gcs_handler.read_file.return_value = "import numpy as np\n\nprint('Hello')"
    
    # Test
    code, md_cells = read_code(mock_gcs_handler, 12345)
    
    # Verify
    assert code == "import numpy as np\n\nprint('Hello')"
    assert md_cells == []


def test_read_code_ipynb(mock_gcs_handler):
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
    code, md_cells = read_code(mock_gcs_handler, 12345)
    
    # Verify
    assert code == "import pandas as pd\ndf = pd.DataFrame()\nprint('Hello')"
    assert md_cells == ["# Title\nDescription", "## Section"]


def test_read_code_nonexistent(mock_gcs_handler):
    """Test reading nonexistent kernel code."""
    # Configure mock
    mock_gcs_handler.get_kernel_path.return_value = None
    
    # Test
    code, md_cells = read_code(mock_gcs_handler, 12345)
    
    # Verify
    assert code is None
    assert md_cells == []


def test_read_code_ipynb_error(mock_gcs_handler):
    """Test error handling with invalid notebook."""
    # Configure mock
    mock_gcs_handler.get_kernel_path.return_value = "0001/234/12345.ipynb"
    mock_gcs_handler.read_file.return_value = "invalid json"
    
    # Test
    code, md_cells = read_code(mock_gcs_handler, 12345)
    
    # Verify
    assert code is None
    assert md_cells == []


@patch('ray.remote')
def test_process_kernel_batch(mock_ray_remote):
    """Test kernel batch processing."""
    # Since ray.remote is complex to mock properly, we'll just check basic function structure
    with patch('src.features.extract_features.GCSFileHandler') as mock_gcs_class:
        with patch('src.features.extract_features.read_code') as mock_read_code:
            with patch('src.features.extract_features.detect_imports') as mock_detect_imports:
                with patch('src.features.extract_features.tag_libraries') as mock_tag_libraries:
                    with patch('src.features.extract_features.tag_architectures') as mock_tag_architectures:
                        with patch('src.features.extract_features.complexity_metrics') as mock_complexity_metrics:
                            with patch('src.features.extract_features.SentenceTransformer') as mock_st_class:
                                with patch('src.features.extract_features.embed_markdown') as mock_embed_markdown:
                                    # Configure mocks
                                    mock_gcs = Mock()
                                    mock_gcs_class.return_value = mock_gcs
                                    
                                    mock_read_code.return_value = ("import numpy as np", ["# Test"])
                                    mock_detect_imports.return_value = {"numpy"}
                                    mock_tag_libraries.return_value = {"data_proc": True}
                                    mock_tag_architectures.return_value = {"CNN": False}
                                    mock_complexity_metrics.return_value = {"loc": 10, "mean_cc": 1, "max_cc": 2}
                                    
                                    model = Mock()
                                    mock_st_class.return_value = model
                                    mock_embed_markdown.return_value = [0.1, 0.2]
                                    
                                    # Call function directly (not through ray)
                                    result = process_kernel_batch._function(
                                        [(12345, 60, "Tesla P100")], 
                                        "test-bucket", 
                                        "all-MiniLM-L6-v2"
                                    )
                                    
                                    # Verify
                                    assert len(result) == 1
                                    assert result[0]["kernel_id"] == 12345
                                    assert result[0]["execSeconds"] == 60
                                    assert result[0]["gpuType"] == "Tesla P100"
                                    assert result[0]["uses_gpu_runtime"] is True
                                    assert result[0]["data_proc"] is True
                                    assert result[0]["CNN"] is False
                                    assert result[0]["loc"] == 10
                                    assert "md_embedding" in result[0]