
"""Setup."""
from setuptools import setup, find_packages

with open("README.MD", "r") as fh:
    long_description = fh.read()

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
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arnaudsm/bigot",
    description="Benchmarking library with Space and Time Complexity estimation",
    include_package_data=True
)
