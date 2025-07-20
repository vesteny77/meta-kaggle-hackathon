"""
Utility functions for accessing code files in the GCS bucket.
"""
from pathlib import Path
import os
from typing import Optional, Tuple
from google.cloud import storage
import requests
from urllib.parse import quote_plus


class GCSFileHandler:
    """Handler for accessing files in Google Cloud Storage."""
    
    def __init__(self, bucket_name: str):
        """Initialize with GCS bucket name."""
        self._user_project = os.getenv("REQUESTER_PAYS_PROJECT")
        if self._user_project:
            self.client = storage.Client(project=os.getenv("REQUESTER_PAYS_PROJECT"))
        else:
        self.client = storage.Client.create_anonymous_client()
        self.bucket = self.client.bucket(bucket_name, user_project=self._user_project)
    
    def list_paths(self, prefix: Optional[str] = None, max_results: int = 10) -> list[str]:
        """List available paths in the bucket."""
        try:
        blobs = self.bucket.list_blobs(prefix=prefix, max_results=max_results)
        return [blob.name for blob in blobs]
        except Exception:
            # Fallback to HTTP API (public buckets only)
            return self._list_paths_via_http(prefix, max_results)

    def _list_paths_via_http(self, prefix: Optional[str], max_results: int) -> list[str]:
        """List objects via public GCS HTTP endpoint (no auth)."""
        base_url = f"https://storage.googleapis.com/storage/v1/b/{self.bucket.name}/o"
        params = {"maxResults": max_results}
        if prefix:
            params["prefix"] = prefix
        params["fields"] = "items(name)"
        if self._user_project:
            params["userProject"] = self._user_project

        try:
            resp = requests.get(base_url, params=params, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            return [item["name"] for item in data.get("items", [])]
        except Exception:
            return []
    
    def read_file(self, gcs_path: str) -> str:
        """Read file content directly without download."""
        blob = self.bucket.blob(gcs_path)
        try:
            if self._user_project:
                return blob.download_as_text(encoding="utf-8", user_project=self._user_project)
            return blob.download_as_text(encoding="utf-8")
        except Exception:
            # Try HTTP fallback for public access
            try:
                url = f"https://storage.googleapis.com/{self.bucket.name}/{quote_plus(gcs_path)}"
                if self._user_project:
                    url += f"?userProject={quote_plus(self._user_project)}"
                resp = requests.get(url, timeout=5)
                resp.raise_for_status()
                return resp.content.decode("utf-8", errors="replace")
        except Exception as e:
                # Fall back to binary via API
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
        # Kaggle stores kernels in a slightly shifted directory scheme:
        #   <digits 2-5>/<digits 6-8>/<kernel_id>.<ext>
        # With a 10-digit zero-padded id string, this translates to:
        #   prefix1 = padded_id[2:6]
        #   prefix2 = padded_id[6:9]
        padded_id = str(kernel_id).zfill(10)
        prefix1 = padded_id[2:6]
        prefix2 = padded_id[6:9]
        
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
                        # Preserve line breaks within the cell
                        code_cells.append("\n".join(source))
                    else:
                        code_cells.append(str(source))
                elif cell.get('cell_type') == 'markdown':
                    source = cell.get('source', [])
                    if isinstance(source, list):
                        md_cells.append("\n".join(source))
                    else:
                        md_cells.append(str(source))
            
            return "\n".join(code_cells), md_cells, file_ext
        except Exception:
            return None, [], file_ext
    
    # Handle Python files (.py)
    elif file_ext in ('.py', '.r', '.rmd'):
        return file_content, [], file_ext
    
    return None, [], file_ext