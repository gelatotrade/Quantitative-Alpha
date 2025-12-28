#!/usr/bin/env python3
"""
Quantitative Alpha: Physics-Based Algorithmic Trading Framework

A unified framework for deterministic chaos, stochastic physics,
and advanced signal processing in quantitative finance.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="quantitative-alpha",
    version="1.0.0",
    author="Quantitative Alpha Research Team",
    author_email="research@quantitative-alpha.com",
    description="Physics-based algorithmic trading framework using chaos theory and stochastic physics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/quantitative-alpha",
    project_urls={
        "Bug Tracker": "https://github.com/your-org/quantitative-alpha/issues",
        "Documentation": "https://github.com/your-org/quantitative-alpha/docs",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    package_dir={"": "."},
    packages=find_packages(where="."),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "advanced": [
            "hmmlearn>=0.2.7",
            "statsmodels>=0.13.0",
            "arch>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "quant-alpha-viz=examples.generate_all_visualizations:main",
        ],
    },
)
