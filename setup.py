"""Setup script for the Meta Kaggle project."""
from setuptools import find_packages, setup

setup(
    name="meta-kaggle-hackathon",
    version="0.1.0",
    packages=find_packages(exclude=["tests"]),
    install_requires=[],
    extras_require={
        "dev": [],
    },
    description="From XGBoost to Transformers: 15 Years of Evolving Kaggle Strategies",
    author="Your Name",
    author_email="your.email@example.com",
)