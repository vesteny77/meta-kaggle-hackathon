"""Utilities for accessing code files stored locally under a two-level hierarchy.

Directory convention (matches Kaggle kernels layout):
    <root>/<prefix1>/<prefix2>/<kernel_id>.<ext>
where
    kernel_id is int (un-padded)
    prefix1 = str(kernel_id).zfill(10)[2:6] (4 digits)
    prefix2 = str(kernel_id).zfill(10)[6:9] (3 digits)

Supported extensions: .py, .ipynb, .r, .rmd
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Tuple, List

ROOT_DIR = Path(os.getenv("LOCAL_CODE_ROOT", "data/raw_code")).resolve()


class LocalFileHandler:
    """Access code files stored on local disk."""

    def __init__(self, root: Path | str | None = None):
        self.root = Path(root).resolve() if root else ROOT_DIR
        if not self.root.exists():
            raise FileNotFoundError(f"Local code root {self.root} does not exist")

    # ---------------------------------------------------------------------
    # Basic helpers
    # ---------------------------------------------------------------------
    def _prefixes(self, kernel_id: int) -> tuple[str, str]:
        padded = str(kernel_id).zfill(10)
        return padded[2:6], padded[6:9]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_kernel_path(self, kernel_id: int) -> Optional[str]:
        """Return relative path (string) to a kernel file if it exists."""
        p1, p2 = self._prefixes(kernel_id)
        for ext in (".py", ".ipynb", ".r", ".rmd"):
            candidate = self.root / p1 / p2 / f"{kernel_id}{ext}"
            if candidate.exists():
                # Return path relative to root to mimic previous behaviour
                return str(candidate)
        return None

    def read_file(self, path: str) -> str:
        return Path(path).read_text(encoding="utf-8", errors="replace")

    # Convenience wrapper used in tests
    def file_exists(self, rel_path: str) -> bool:
        return (self.root / rel_path).exists()


# -------------------------------------------------------------------------
# Helper: read kernel code
# -------------------------------------------------------------------------

def read_kernel_code(handler: LocalFileHandler, kernel_id: int) -> Tuple[Optional[str], List[str], Optional[str]]:
    file_path = handler.get_kernel_path(kernel_id)
    if not file_path:
        return None, [], None

    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    try:
        file_content = handler.read_file(file_path)
    except Exception:
        return None, [], ext

    if ext == ".ipynb":
        try:
            nb = json.loads(file_content)
            code_cells, md_cells = [], []
            for cell in nb.get("cells", []):
                if cell.get("cell_type") == "code":
                    src = cell.get("source", [])
                    code_cells.append("\n".join(src) if isinstance(src, list) else str(src))
                elif cell.get("cell_type") == "markdown":
                    src = cell.get("source", [])
                    md_cells.append("\n".join(src) if isinstance(src, list) else str(src))
            return "\n".join(code_cells), md_cells, ext
        except Exception:
            return None, [], ext

    # plain text files
    return file_content, [], ext 