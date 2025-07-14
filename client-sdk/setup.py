"""
Setup configuration for quantum-rerank-client package.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Python client for QuantumRerank API"

setup(
    name="quantum-rerank-client",
    version="1.0.0",
    description="Python client for QuantumRerank quantum-enhanced semantic similarity API",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="QuantumRerank Team",
    author_email="support@quantumrerank.ai",
    url="https://github.com/quantumrerank/python-client",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords="quantum computing, semantic similarity, machine learning, nlp, reranking",
    python_requires=">=3.7",
    install_requires=[
        "requests>=2.25.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-mock>=3.0.0",
            "requests-mock>=1.9.0",
            "flake8>=4.0.0",
            "black>=22.0.0",
        ]
    },
    project_urls={
        "Bug Reports": "https://github.com/quantumrerank/python-client/issues",
        "Documentation": "https://docs.quantumrerank.ai/python-client",
        "Source": "https://github.com/quantumrerank/python-client",
    },
)