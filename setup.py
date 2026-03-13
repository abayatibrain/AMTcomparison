#!/usr/bin/env python
# coding: utf-8
"""
Setup configuration for AMTComparison package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8") if (this_directory / "README.md").exists() else ""

setup(
    name="AMTComparison",
    version="2.0.0",
    description="Advanced Multi-Neuron Analysis Comparison Tool for neuron morphology and dynamics analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AMT Development Team",
    author_email="contact@amtanalysis.dev",
    url="https://github.com/abayatibrain/AMTcomparison",
    license="MIT",
    py_modules=["AMTComparison"],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "matplotlib>=3.2.0",
        "seaborn>=0.11.0",
        "scipy>=1.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=20.8b1",
            "flake8>=3.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "amt-compare=AMTComparison:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords=[
        "neuron",
        "morphology",
        "analysis",
        "comparison",
        "microscopy",
        "comparative-analysis",
        "statistical-testing",
    ],
)
