#!/bin/bash
# Phased setup script for conda environment to help diagnose issues

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install Miniconda or Anaconda first."
    echo "Visit https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Step 1: Create a basic environment with just Python
echo "Step 1: Creating base conda environment with Python 3.11..."
conda create -n meta-kaggle python=3.11 -y

# Step 2: Activate the environment
echo "Step 2: Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate meta-kaggle

# Step 3: Add core conda packages one by one
echo "Step 3: Installing core conda packages..."
conda_packages=(
    "pip"
    "polars>=0.19.0"
    "duckdb>=0.9.0"
    "pyarrow>=13.0.0"
    "pandas>=2.0.0"
    "networkx>=3.0"
    "plotly>=5.10.0"
    "dash>=2.8.0"
    "scipy>=1.10.0"
)

for package in "${conda_packages[@]}"; do
    echo "Installing $package..."
    conda install -c conda-forge "$package" -y
    if [ $? -ne 0 ]; then
        echo "Warning: Failed to install $package from conda-forge, trying defaults..."
        conda install -c defaults "$package" -y
        if [ $? -ne 0 ]; then
            echo "Error: Failed to install $package. Please check package availability."
            echo "You can continue and install remaining packages."
            read -p "Continue? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "Setup aborted by user."
                exit 1
            fi
        fi
    fi
done

# Step 4: Install ray-core (this can be problematic sometimes)
echo "Step 4: Installing ray-core..."
conda install -c conda-forge ray-core>=2.6.0 -y
if [ $? -ne 0 ]; then
    echo "Warning: Failed to install ray-core. This package sometimes has conflicts."
    echo "You can try installing it later or use pip to install it."
fi

# Step 5: Install dev tools
echo "Step 5: Installing development tools..."
conda install -c conda-forge pytest>=7.0.0 black>=23.0.0 ruff>=0.0.265 mypy>=1.0.0 pre-commit>=3.0.0 -y

# Step 6: Install pip packages
echo "Step 6: Installing pip packages..."
pip_packages=(
    "sentence-transformers>=2.2.0"
    "bertopic>=0.14.0"
    "lifelines>=0.27.0"
    "statsmodels>=0.13.0"
    "prophet>=1.1.0"
    "radon>=5.1.0"
    "pymannkendall>=1.4"
    "kedro>=0.18.0"
    "pytest-cov>=4.0.0"
)

for package in "${pip_packages[@]}"; do
    echo "Installing $package..."
    pip install "$package"
    if [ $? -ne 0 ]; then
        echo "Warning: Failed to install $package through pip."
        echo "You can continue and install remaining packages."
        read -p "Continue? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Setup aborted by user."
            exit 1
        fi
    fi
done

# Step 7: Verify installation
echo "Step 7: Verifying installation..."
conda list

echo ""
echo "Environment setup complete! If there were any failures, check the logs above."
echo ""
echo "The environment is already activated. To deactivate when finished, run:"
echo "conda deactivate"