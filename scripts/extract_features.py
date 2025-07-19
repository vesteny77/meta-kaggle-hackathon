#!/usr/bin/env python
"""
Extract features from code files.
"""
import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Switch to GCS for accessing kernel files
# from src.features.path_utils import find_code_file
from src.features.extract_features import main as run_extraction

# Directory paths
META_PATH = Path("data/intermediate/kernel_bigjoin_clean.parquet")
OUTPUT_PATH = Path("data/processed/kernel_features.parquet")
CODE_DIR = Path("data/raw_code")


def read_code(file_path: Path) -> Tuple[str, List[str]]:
    """
    Read code from a file, handling different file types.

    Args:
        file_path: Path to the code file

    Returns:
        tuple: (code_string, markdown_list)
            - code_string: Concatenated code content
            - markdown_list: List of markdown cells (for notebooks)
    """
    if file_path.suffix.lower() == ".py":
        code = file_path.read_text(encoding="utf-8", errors="ignore")
        return code, []

    elif file_path.suffix.lower() == ".ipynb":
        try:
            nb = json.loads(file_path.read_text(encoding="utf-8", errors="ignore"))
            code_cells, md_cells = [], []

            for cell in nb.get("cells", []):
                if cell.get("cell_type") == "code":
                    code_cells.append("".join(cell.get("source", [])))
                elif cell.get("cell_type") == "markdown":
                    md_cells.append("".join(cell.get("source", [])))

            return "\n".join(code_cells), md_cells

        except (json.JSONDecodeError, AttributeError):
            print(f"Error parsing notebook: {file_path}", file=sys.stderr)
            return "", []

    elif file_path.suffix.lower() in (".r", ".rmd"):
        code = file_path.read_text(encoding="utf-8", errors="ignore")
        return code, []

    else:
        print(f"Unsupported file type: {file_path.suffix}", file=sys.stderr)
        return "", []


def detect_imports(code: str) -> Dict[str, bool]:
    """
    Detect imports in Python code.

    Args:
        code: Python code as string

    Returns:
        dict: Dictionary of library flags
    """
    libraries = {
        "gbdt": {"xgboost", "lightgbm", "catboost"},
        "dl_pytorch": {"torch", "pytorch_lightning", "torchvision"},
        "dl_tf": {"tensorflow", "keras", "tf"},
        "auto_ml": {"autogluon", "autosklearn", "pycaret", "tpot"},
        "gpu_accel": {"cudf", "cupy", "rapids", "jax"},
    }

    imports = set()

    # Try to parse with ast
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    imports.add(n.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split(".")[0])
    except SyntaxError:
        # Fallback to regex for non-Python or broken code
        import_pattern = r"(?:import|from)\s+([a-zA-Z0-9_.]+)"
        matches = re.findall(import_pattern, code)
        for match in matches:
            imports.add(match.split(".")[0])

    # Map imports to library flags
    flags = {}
    for lib_name, lib_modules in libraries.items():
        flags[lib_name] = any(imp in lib_modules for imp in imports)

    return flags


def process_kernel(
    kernel_id: int, meta_row: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Process a single kernel file.

    Args:
        kernel_id: Kernel ID
        meta_row: Metadata for the kernel

    Returns:
        dict: Features extracted from the kernel, or None if the file is not found
    """
    try:
        file_path = find_code_file(kernel_id, CODE_DIR)
    except FileNotFoundError:
        print(f"Could not find code file for kernel {kernel_id}", file=sys.stderr)
        return None

    # Read the code file
    code, md_cells = read_code(file_path)
    if not code:
        return None

    # Extract features
    features = {
        "kernel_id": kernel_id,
        "file_type": file_path.suffix.lower(),
        "code_length": len(code),
        "line_count": len(code.splitlines()),
        "markdown_count": len(md_cells),
    }

    # Add metadata
    features.update(
        {
            "exec_seconds": meta_row.get("execSeconds"),
            "gpu_type": meta_row.get("gpuType", "None"),
            "uses_gpu": meta_row.get("gpuType") is not None
            and meta_row.get("gpuType") != "None",
        }
    )

    # Add library flags for Python files
    if file_path.suffix.lower() in (".py", ".ipynb"):
        features.update(detect_imports(code))

    return features


def main():
    """Main function to extract features from code files."""
    parser = argparse.ArgumentParser(description="Extract features from code files")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of kernels to process (0 for all)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument(
        "--gcs-bucket",
        type=str,
        default="kaggle-meta-kaggle-code-downloads",
        help="GCS bucket containing kernel files",
    )
    args = parser.parse_args()

    # Use the new implementation that works with GCS
    print("Starting feature extraction from GCS bucket...")
    try:
        run_extraction()
        print("Feature extraction completed successfully")
    except Exception as e:
        print(f"Error during feature extraction: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
