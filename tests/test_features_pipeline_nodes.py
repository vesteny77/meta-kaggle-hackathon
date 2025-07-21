"""
Tests for feature extraction pipeline nodes.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import polars as pl
import tempfile

from src.pipeline.features.nodes import (
    extract_features_node,
    topic_modeling_node,
    merge_features_node
)


@pytest.fixture
def sample_params():
    """Sample parameters for pipeline nodes."""
    return {
        "gcs_bucket": "test-bucket",
        "output_dir": "data/intermediate",
        "sample_size": 10,
        "batch_size": 5,
        "embedding_model": "test-model",
        "num_cpus": 2,
        "only_commits": True,
        "processed_dir": "data/processed",
        "keep_embeddings": False,
        "num_topics": 5
    }


@pytest.fixture
def sample_metadata(tmpdir):
    """Create sample metadata parquet file."""
    df = pl.DataFrame({
        "kernel_version_id": [1, 2, 3, 4, 5],
        "kernel_id": [101, 102, 103, 104, 105],
        "execSeconds": [10, 20, 30, 40, 50],
        "gpuType": ["None", "Tesla P100", "None", "Tesla V100", "None"],
        "is_commit": [True, True, False, True, True]
    })
    
    file_path = Path(tmpdir) / "metadata.parquet"
    df.write_parquet(file_path)
    
    return str(file_path)


@pytest.fixture
def sample_features(tmpdir):
    """Create sample features parquet file."""
    df = pl.DataFrame({
        "kernel_id": [1, 2, 3, 4, 5],
        "execSeconds": [10, 20, 30, 40, 50],
        "gpuType": ["None", "Tesla P100", "None", "Tesla V100", "None"],
        "uses_gpu_runtime": [False, True, False, True, False],
        "dl_pytorch": [True, False, False, True, False],
        "md_embedding": [[0.1, 0.2], [0.3, 0.4], None, [0.5, 0.6], None]
    })
    
    file_path = Path(tmpdir) / "features.parquet"
    df.write_parquet(file_path)
    
    return str(file_path)


@pytest.fixture
def sample_topics(tmpdir):
    """Create sample topics parquet file."""
    df = pl.DataFrame({
        "kernel_id": [1, 2, 4],
        "topic_id": [0, 1, 2],
        "topic_prob": [0.8, 0.9, 0.7],
        "topic_name": ["Topic 0", "Topic 1", "Topic 2"]
    })
    
    file_path = Path(tmpdir) / "topics.parquet"
    df.write_parquet(file_path)
    
    return str(file_path)


@patch('ray.init')
@patch('ray.remote')
@patch('ray.get')
def test_extract_features_node(mock_ray_get, mock_ray_remote, mock_ray_init, sample_metadata, sample_params):
    """Test the extract_features_node."""
    # Set up ray mock
    mock_future = Mock()
    mock_ray_remote.return_value.remote.return_value = mock_future
    mock_ray_get.return_value = [[
        {"kernel_id": 1, "execSeconds": 10, "data_proc": True},
        {"kernel_id": 2, "execSeconds": 20, "dl_pytorch": True}
    ]]
    
    # Execute function
    output_path = extract_features_node(sample_params)
    
    # Verify
    assert isinstance(output_path, Path)
    assert output_path.name == "kernel_features_raw.parquet"
    
    # Check ray init was called
    mock_ray_init.assert_called_once_with(num_cpus=2, ignore_reinit_error=True)
    
    # Check process_kernel_batch was called via ray
    mock_ray_remote.return_value.remote.assert_called_once()
    mock_ray_get.assert_called_once_with([mock_future])


@patch('src.pipeline.features.nodes.BERTopic')
def test_topic_modeling_node(mock_bertopic_class, sample_features, sample_params):
    """Test the topic_modeling_node."""
    # Set up mock
    mock_bertopic = Mock()
    mock_bertopic.fit_transform.return_value = ([0, 1, 2], [0.8, 0.9, 0.7])
    mock_bertopic.get_topic_info.return_value = pl.DataFrame({
        "Topic": [0, 1, 2],
        "Name": ["Topic 0", "Topic 1", "Topic 2"]
    }).to_pandas()
    mock_bertopic_class.return_value = mock_bertopic
    
    # Execute function
    with patch('pickle.dump') as mock_pickle:
        output_path = topic_modeling_node(sample_features, sample_params)
    
    # Verify
    assert isinstance(output_path, Path)
    assert output_path.name == "markdown_topics.parquet"
    
    # Check BERTopic was created and used
    mock_bertopic_class.assert_called_once_with(verbose=True, nr_topics=5)
    mock_bertopic.fit_transform.assert_called_once()
    mock_pickle.assert_called_once()


def test_topic_modeling_node_no_embeddings(sample_params):
    """Test topic_modeling_node with no embeddings."""
    # Create features with no embeddings
    with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
        df = pl.DataFrame({
            "kernel_id": [1, 2, 3],
            "execSeconds": [10, 20, 30]
        })
        df.write_parquet(tmp.name)
        
        # Execute function should raise error
        with pytest.raises(ValueError, match="No markdown embeddings found"):
            topic_modeling_node(tmp.name, sample_params)


def test_merge_features_node(sample_features, sample_topics, sample_metadata, sample_params):
    """Test the merge_features_node."""
    # Execute function
    output_path = merge_features_node(sample_features, sample_topics, sample_params)
    
    # Verify
    assert isinstance(output_path, Path)
    assert output_path.name == "kernel_features.parquet"
    
    # Read the output file to check content
    result_df = pl.read_parquet(output_path)
    
    # Check that md_embedding column was dropped
    assert "md_embedding" not in result_df.columns
    
    # Check that topic columns were added
    assert "topic_id" in result_df.columns
    assert "topic_name" in result_df.columns


def test_merge_features_node_keep_embeddings(sample_features, sample_topics, sample_metadata):
    """Test merge_features_node with keep_embeddings=True."""
    # Modify params to keep embeddings
    params = {
        "processed_dir": "data/processed",
        "keep_embeddings": True
    }
    
    # Execute function
    output_path = merge_features_node(sample_features, sample_topics, params)
    
    # Verify
    result_df = pl.read_parquet(output_path)
    
    # Check that md_embedding column was not dropped
    assert "md_embedding" in result_df.columns


def test_merge_features_node_no_topics(sample_features, sample_params):
    """Test merge_features_node without topics."""
    # Execute function with None for topics path
    output_path = merge_features_node(sample_features, None, sample_params)
    
    # Verify
    assert isinstance(output_path, Path)
    
    # Read the output file to check content
    result_df = pl.read_parquet(output_path)
    
    # Check that topic columns were not added
    assert "topic_id" not in result_df.columns
    assert "topic_name" not in result_df.columns