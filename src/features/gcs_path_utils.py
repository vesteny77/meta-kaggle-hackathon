"""
Utility functions for accessing code files in the GCS bucket.
"""
from pathlib import Path
import os
from typing import Optional, Tuple
from google.cloud import storage


class GCSFileHandler:
    """Handler for accessing files in Google Cloud Storage."""
    
    def __init__(self, bucket_name: str):
        """Initialize with GCS bucket name."""
        self.client = storage.Client.create_anonymous_client()
        self.bucket = self.client.bucket(bucket_name)
    
    def list_paths(self, prefix: Optional[str] = None, max_results: int = 10) -> list[str]:
        """List available paths in the bucket."""
        blobs = self.bucket.list_blobs(prefix=prefix, max_results=max_results)
        return [blob.name for blob in blobs]
    
    def read_file(self, gcs_path: str) -> str:
        """Read file content directly without download."""
        blob = self.bucket.blob(gcs_path)
        try:
            return blob.download_as_text(encoding="utf-8")
        except Exception as e:
            # Fall back to binary for non-text files
            try:
                binary_content = blob.download_as_bytes()
                return binary_content.decode('utf-8', errors='replace')
            except Exception:
                raise IOError(f"Could not read file {gcs_path}: {str(e)}")
    
    def file_exists(self, gcs_path: str) -> bool:
        """Check if a file exists in the bucket."""
        blob = self.bucket.blob(gcs_path)
        return blob.exists()
    
    def get_kernel_path(self, kernel_id: int) -> Optional[str]:
        """Generate path for a kernel ID."""
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
            
        # Check for .r file
        r_path = f"{prefix1}/{prefix2}/{kernel_id}.r"
        if self.file_exists(r_path):
            return r_path
            
        # Check for .rmd file
        rmd_path = f"{prefix1}/{prefix2}/{kernel_id}.rmd"
        if self.file_exists(rmd_path):
            return rmd_path
        
        return None


def read_kernel_code(handler: GCSFileHandler, kernel_id: int) -> Tuple[Optional[str], list[str], Optional[str]]:
    """
    Read code from a kernel file in GCS.
    
    Args:
        handler: GCS file handler
        kernel_id: Kernel ID
        
    Returns:
        tuple: (code_str, markdown_list, file_ext)
    """
    file_path = handler.get_kernel_path(kernel_id)
    
    if not file_path:
        return None, [], None
    
    try:
        file_content = handler.read_file(file_path)
        file_ext = os.path.splitext(file_path)[1].lower()
    except Exception:
        return None, [], None
    
    # Handle Jupyter notebooks (.ipynb)
    if file_ext == '.ipynb':
        try:
            import json
            nb = json.loads(file_content)
            code_cells = []
            md_cells = []
            
            for cell in nb.get('cells', []):
                if cell.get('cell_type') == 'code':
                    source = cell.get('source', [])
                    if isinstance(source, list):
                        code_cells.append("".join(source))
                    else:
                        code_cells.append(str(source))
                elif cell.get('cell_type') == 'markdown':
                    source = cell.get('source', [])
                    if isinstance(source, list):
                        md_cells.append("".join(source))
                    else:
                        md_cells.append(str(source))
            
            return "\n".join(code_cells), md_cells, file_ext
        except Exception:
            return None, [], file_ext
    
    # Handle Python files (.py)
    elif file_ext in ('.py', '.r', '.rmd'):
        return file_content, [], file_ext
    
    return None, [], file_ext