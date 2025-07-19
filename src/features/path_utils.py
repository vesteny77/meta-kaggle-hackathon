"""
Utility functions for resolving paths to code files in the raw_code directory.
"""

import os
from pathlib import Path


def get_code_file_path(kernel_id: int, code_dir: Path = Path("data/raw_code")) -> Path:
    """
    Resolve the path to a code file based on the kernel ID.
    The directory structure is:
    data/raw_code/XXXX/YYY/kernel_id.{py|r|ipynb}

    where:
    - XXXX is the first 4 digits of the kernel ID (padded with leading zeros)
    - YYY is the next 3 digits of the kernel ID (padded with leading zeros)

    Args:
        kernel_id: The ID of the kernel
        code_dir: Base directory for code files (default: data/raw_code)

    Returns:
        Path: Full path to the code file (without extension)
    """
    try:
        kernel_id = int(kernel_id)
    except (ValueError, TypeError):
        raise TypeError("kernel_id must be an integer or a string that can be converted to an integer")
    # Convert to string
    id_str = str(kernel_id)

    # Extract the directory components. Use modulo arithmetic for distribution.
    dir1 = str(kernel_id % 1000).zfill(3)
    dir2 = str((kernel_id // 1000) % 1000).zfill(3)
    
    # Construct the directory path
    dir_path = code_dir / dir2 / dir1

    # Return the path without extension (caller will need to check extensions)
    return dir_path / id_str


def find_code_file(kernel_id: int, code_dir: Path = Path("data/raw_code")) -> Path:
    """
    Find a code file by kernel ID, checking for common extensions.

    Args:
        kernel_id: The ID of the kernel
        code_dir: Base directory for code files (default: data/raw_code)

    Returns:
        Path: Path to the found code file

    Raises:
        FileNotFoundError: If no matching file is found
    """
    base_path = get_code_file_path(kernel_id, code_dir)

    # Check for common extensions
    extensions = [".py", ".ipynb", ".r", ".R"]

    for ext in extensions:
        file_path = Path(f"{base_path}{ext}")
        if file_path.exists():
            return file_path

    raise FileNotFoundError(f"No code file found for kernel ID {kernel_id}")


def list_code_files(code_dir: Path = Path("data/raw_code")) -> list[Path]:
    """
    List all code files in the raw_code directory.

    Args:
        code_dir: Base directory for code files (default: data/raw_code)

    Returns:
        list[Path]: List of paths to code files
    """
    files = []

    # Walk the directory tree
    for root, _, filenames in os.walk(code_dir):
        for filename in filenames:
            if filename.endswith((".py", ".ipynb", ".r", ".R")):
                files.append(Path(root) / filename)

    return files
