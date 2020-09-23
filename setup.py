
"""Setup."""
from setuptools import setup, find_packages

setup(
    name="bigot",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "memory-profiler",
        "pandas",
        "plotly",
    ],
    author="Arnaud de Saint Meloir",
    author_email="arnaud.desaintmeloir@gmail.com",
    description="Benchmarking library with Space and Time Complexity estimation",
    include_package_data=True
)
