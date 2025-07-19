"""Project settings."""

from pathlib import Path

# This is the location of the project configuration directory
CONF_SOURCE = "conf"

# The location of the project root directory.
# If you're running this from a script in the project directory,
# this should work automatically.
# Otherwise, specify the absolute path to the project root.
PROJECT_ROOT = Path(__file__).parents[1]  # Going up from src to the project root

# Subdirectories for various data types
CATALOG_PATH = "catalog"
CREDENTIALS_PATH = "credentials"
PARAMETERS_PATH = "parameters"
LOGGING_PATH = "logging"
