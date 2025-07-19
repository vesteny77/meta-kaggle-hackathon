"""Setup script for the Meta Kaggle project."""

from setuptools import find_packages, setup

setup(
    name="meta-kaggle-hackathon",
    version="0.1.0",
    packages=find_packages(exclude=["tests"]),
    install_requires=[
        "polars>=0.19.0",
        "duckdb>=0.9.0",
        "pyarrow>=13.0.0",
        "pandas>=2.0.0",
        "kedro>=0.18.0",
        "networkx>=3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "ruff>=0.0.265",
        ],
    },
    python_requires=">=3.11",
    description="From XGBoost to Transformers: 15 Years of Evolving Kaggle Strategies",
    author="Your Name",
    author_email="your.email@example.com",
)
