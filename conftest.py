"""
Root conftest.py for pytest configuration.
"""
import sys
import os
from pathlib import Path

# Add the project root directory to the Python path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))