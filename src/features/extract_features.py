#!/usr/bin/env python
"""
Feature extraction module for Meta Kaggle kernels.

This module analyzes kernel code files from the Google Cloud Storage bucket to extract:
- Library usage patterns
- Model architecture patterns
- Code complexity metrics
- Markdown embeddings for topic modeling

The outputs are stored in Parquet format for downstream analysis.
"""
import os
import json
import re
import ast
import polars as pl
import ray
from radon.complexity import cc_visit
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
from google.cloud import storage
import tiktoken
import logging
from tqdm import tqdm
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
METADATA = "data/intermediate/kernel_bigjoin_clean.parquet"
GCS_BUCKET_NAME = "kaggle-meta-kaggle-code-downloads"
OUTPUT_DIR = Path("data/intermediate")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define taxonomies for library tagging
LIB_TAGS = {
    "gbdt": {"xgboost", "lightgbm", "catboost"},
    "dl_pytorch": {"torch", "pytorch-lightning", "torchvision", "fastai"},
    "dl_tf": {"tensorflow", "keras", "tf.keras"},
    "auto_ml": {"autogluon", "autosklearn", "pycaret", "tpot", "h2o"},
    "gpu_accel": {"cudf", "cupy", "rapids", "jax"},
    "vision": {"opencv", "cv2", "pillow", "PIL", "albumentations", "imgaug"},
    "nlp": {"nltk", "spacy", "transformers", "huggingface", "gensim", "allennlp"},
    "data_proc": {"pandas", "polars", "numpy", "dask", "vaex"},
    "viz": {"matplotlib", "plotly", "seaborn", "bokeh", "altair"}
}

# Regex patterns for architecture detection
ARCH_REGEX = {
    "ResNet": r"[Rr]esNet\d{1,3}",
    "EfficientNet": r"EfficientNet[Bb]\d",
    "ViT": r"VisionTransformer|ViT[Bb]",
    "Transformer": r"(nn\.Transformer|TransformerEncoderLayer)",
    "BERT": r"\bBertModel\b|\bbert-base|\broberta\b",
    "GPT": r"\bGPT2?Model\b|gpt[-_]?2",
    "LSTM": r"\bLSTM\b",
    "GRU": r"\bGRU\b",
    "CNN": r"\bConv[12]d\b|\bConvolution[12]d\b",
    "Attention": r"Attention|MultiHead|SelfAttention",
    "UNet": r"\bUNet\b|\bU-Net\b",
    "GAN": r"\bGAN\b|\bCycleGAN\b|\bDCGAN\b|\bStyleGAN\b",
}


class GCSFileHandler:
    """Handler for accessing files in Google Cloud Storage."""
    
    def __init__(self, bucket_name):
        """Initialize with GCS bucket name."""
        self.client = storage.Client.create_anonymous_client()
        self.bucket = self.client.bucket(bucket_name)
        
    def list_paths(self, prefix=None):
        """List available paths in the bucket."""
        blobs = self.bucket.list_blobs(prefix=prefix, max_results=10)
        return [blob.name for blob in blobs]
    
    def list_sample_paths(self, prefix=None, limit=5):
        """List sample paths to understand structure."""
        blobs = self.bucket.list_blobs(prefix=prefix, max_results=limit)
        return [blob.name for blob in blobs]
    
    def download_to_tempfile(self, gcs_path):
        """Download a file to a temporary location."""
        blob = self.bucket.blob(gcs_path)
        _, temp_filename = tempfile.mkstemp()
        blob.download_to_filename(temp_filename)
        return temp_filename
    
    def read_file(self, gcs_path):
        """Read file content directly without download."""
        blob = self.bucket.blob(gcs_path)
        return blob.download_as_text(encoding="utf-8")
    
    def file_exists(self, gcs_path):
        """Check if a file exists in the bucket."""
        blob = self.bucket.blob(gcs_path)
        return blob.exists()
    
    def get_kernel_path(self, kernel_id):
        """Generate potential file paths for a kernel ID."""
        # Determine the prefix dirs based on kernel ID
        padded_id = str(kernel_id).zfill(10)
        prefix1 = padded_id[:4]
        prefix2 = padded_id[4:7]
        
        # Check for .py file
        py_path = f"{prefix1}/{prefix2}/{kernel_id}.py"
        if self.file_exists(py_path):
            return py_path
            
        # Check for .ipynb file
        ipynb_path = f"{prefix1}/{prefix2}/{kernel_id}.ipynb"
        if self.file_exists(ipynb_path):
            return ipynb_path
        
        # Return None if neither exists
        return None


def read_code(gcs_handler, kernel_id):
    """
    Read code from a kernel file in GCS.
    
    Args:
        gcs_handler: GCS file handler
        kernel_id: Kernel ID
        
    Returns:
        tuple: (code_str, markdown_list)
    """
    file_path = gcs_handler.get_kernel_path(kernel_id)
    
    if not file_path:
        return None, []
    
    file_content = gcs_handler.read_file(file_path)
    
    # Handle Jupyter notebooks (.ipynb)
    if file_path.endswith('.ipynb'):
        try:
            nb = json.loads(file_content)
            code_cells = []
            md_cells = []
            
            for cell in nb.get('cells', []):
                if cell.get('cell_type') == 'code':
                    code_cells.append("".join(cell.get('source', [])))
                elif cell.get('cell_type') == 'markdown':
                    md_cells.append("".join(cell.get('source', [])))
            
            return "\n".join(code_cells), md_cells
        except Exception as e:
            logger.error(f"Error parsing notebook {kernel_id}: {str(e)}")
            return None, []
    
    # Handle Python files (.py)
    elif file_path.endswith('.py'):
        return file_content, []
    
    return None, []


def detect_imports(code_str):
    """
    Extract imported module names from Python code.
    
    Args:
        code_str: Python code as string
        
    Returns:
        set: Set of imported module names
    """
    if not code_str:
        return set()
    
    imports = set()
    
    # Try using AST for reliable parsing
    try:
        tree = ast.parse(code_str)
        for node in ast.walk(tree):
            # Handle 'import numpy as np' style
            if isinstance(node, ast.Import):
                for name in node.names:
                    # Get the base module name (e.g., 'numpy' from 'numpy.random')
                    imports.add(name.name.split('.')[0])
            
            # Handle 'from tensorflow import keras' style
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
    except SyntaxError:
        # Fall back to regex for files with syntax errors
        import_regex = r"^\s*import\s+([a-zA-Z0-9_\.]+)"
        from_regex = r"^\s*from\s+([a-zA-Z0-9_\.]+)\s+import"
        
        for line in code_str.split('\n'):
            # Check for 'import' statements
            match = re.match(import_regex, line)
            if match:
                imports.add(match.group(1).split('.')[0])
                continue
                
            # Check for 'from ... import' statements
            match = re.match(from_regex, line)
            if match:
                imports.add(match.group(1).split('.')[0])
    
    return imports


def tag_libraries(imports):
    """
    Tag imported libraries based on predefined categories.
    
    Args:
        imports: Set of imported module names
        
    Returns:
        dict: Dictionary of library category flags
    """
    return {category: bool(libs.intersection(imports)) for category, libs in LIB_TAGS.items()}


def tag_architectures(code_str):
    """
    Detect ML architecture patterns in code.
    
    Args:
        code_str: Python code as string
        
    Returns:
        dict: Dictionary of architecture flags
    """
    if not code_str:
        return {arch: False for arch in ARCH_REGEX}
    
    return {arch: bool(re.search(pattern, code_str)) for arch, pattern in ARCH_REGEX.items()}


def complexity_metrics(code_str):
    """
    Calculate code complexity metrics.
    
    Args:
        code_str: Python code as string
        
    Returns:
        dict: Dictionary of complexity metrics
    """
    if not code_str:
        return {"loc": 0, "mean_cc": 0, "max_cc": 0}
    
    # Calculate lines of code
    loc = len(code_str.splitlines())
    
    # Calculate cyclomatic complexity
    try:
        cc_scores = [cc.complexity for cc in cc_visit(code_str)]
        mean_cc = sum(cc_scores) / len(cc_scores) if cc_scores else 0
        max_cc = max(cc_scores) if cc_scores else 0
    except Exception:
        mean_cc = max_cc = 0
    
    return {"loc": loc, "mean_cc": mean_cc, "max_cc": max_cc}


def embed_markdown(md_list, model):
    """
    Create embeddings for markdown text.
    
    Args:
        md_list: List of markdown text strings
        model: SentenceTransformer model
        
    Returns:
        list: Embedding vector or None
    """
    if not md_list:
        return None
        
    # Concatenate markdown and limit to first 8K characters to avoid too long inputs
    text = " ".join(md_list)[:8192]
    
    try:
        return model.encode(text).tolist() if text else None
    except Exception as e:
        logger.error(f"Error embedding markdown: {str(e)}")
        return None


@ray.remote
def process_kernel_batch(kernel_batch, gcs_bucket_name, model_name=None):
    """
    Process a batch of kernels (Ray task).
    
    Args:
        kernel_batch: List of kernel IDs and metadata
        gcs_bucket_name: GCS bucket name
        model_name: Optional name of SentenceTransformer model
        
    Returns:
        list: List of feature dictionaries
    """
    gcs_handler = GCSFileHandler(gcs_bucket_name)
    
    # Initialize embedding model if needed
    model = None
    if model_name:
        try:
            model = SentenceTransformer(model_name)
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
    
    results = []
    for kernel_id, exec_sec, gpu_type in kernel_batch:
        # Read code files from GCS
        code_str, md_cells = read_code(gcs_handler, kernel_id)
        
        if code_str is None:
            continue
            
        # Extract features
        imports = detect_imports(code_str)
        features = {
            "kernel_id": int(kernel_id),
            "execSeconds": exec_sec,
            "gpuType": gpu_type or "None",
            "uses_gpu_runtime": gpu_type not in (None, "None"),
            **tag_libraries(imports),
            **tag_architectures(code_str),
            **complexity_metrics(code_str),
        }
        
        # Add markdown embeddings if model is available
        if model and md_cells:
            emb = embed_markdown(md_cells, model)
            if emb is not None:
                features["md_embedding"] = emb
        
        results.append(features)
    
    return results


def main():
    """Extract features from kernel code files."""
    start_time = time.time()
    logger.info("Starting feature extraction module")
    
    # Initialize Ray with auto-detected resources
    ray.init(num_cpus=os.cpu_count())
    
    # Create GCS handler to explore bucket structure
    gcs_handler = GCSFileHandler(GCS_BUCKET_NAME)
    
    # List some files to understand structure
    logger.info("Exploring GCS bucket structure...")
    sample_paths = gcs_handler.list_sample_paths(limit=10)
    logger.info(f"Sample paths in GCS bucket: {sample_paths}")
    
    # Load metadata from parquet
    logger.info(f"Loading kernel metadata from {METADATA}")
    try:
        meta_df = pl.read_parquet(METADATA).filter(pl.col("isCommit") == True)
        logger.info(f"Loaded metadata for {len(meta_df)} kernels")
    except Exception as e:
        logger.error(f"Error loading metadata: {str(e)}")
        meta_df = None
        
    if meta_df is None or len(meta_df) == 0:
        logger.error("No metadata available, creating sample data for development")
        # Create sample data for development
        meta_df = pl.DataFrame({
            "kernel_id": [1000, 1001, 1002],
            "execSeconds": [100, 200, 300],
            "gpuType": ["None", "Tesla P100", "None"]
        })
    
    # Extract kernels to process
    kernels_data = meta_df.select(["kernel_id", "execSeconds", "gpuType"]).rows()
    
    # Process a sample of kernels for testing
    sample_size = min(1000, len(kernels_data))
    kernels_to_process = kernels_data[:sample_size]
    
    logger.info(f"Processing {len(kernels_to_process)} kernels")
    
    # Split data into batches for Ray
    batch_size = 100
    batches = [kernels_to_process[i:i + batch_size] 
               for i in range(0, len(kernels_to_process), batch_size)]
    
    # Process batches in parallel with Ray
    logger.info(f"Submitting {len(batches)} batches to Ray workers")
    model_name = "all-MiniLM-L6-v2"  # Small and fast model
    futures = [process_kernel_batch.remote(batch, GCS_BUCKET_NAME, model_name) 
               for batch in batches]
    
    # Collect results
    logger.info("Processing batches...")
    all_results = []
    for batch_result in tqdm(ray.get(futures), total=len(futures)):
        all_results.extend(batch_result)
    
    # Convert results to DataFrame
    logger.info(f"Creating DataFrame from {len(all_results)} processed kernels")
    result_df = pl.DataFrame(all_results)
    
    # Save raw features
    output_path = OUTPUT_DIR / "kernel_features_raw.parquet"
    logger.info(f"Saving features to {output_path}")
    result_df.write_parquet(output_path)
    
    # Print summary
    elapsed = time.time() - start_time
    logger.info(f"Feature extraction completed in {elapsed:.2f} seconds")
    logger.info(f"Processed {len(result_df)} kernels")
    logger.info(f"Output saved to {output_path}")
    
    # Shut down Ray
    ray.shutdown()


if __name__ == "__main__":
    main()