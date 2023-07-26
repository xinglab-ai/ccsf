from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="ccsf",
    version="1.0.2",
    author="Md Tauhidul Islam",
    author_email="tauhid@stanford.edu",
    description="Leveraging cell-cell similarity from gene expression data for high-performance spatial and temporal cellular mappings.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xinglab-ai/ccsf",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=open("requirements.txt").readlines(),
    include_package_data=True,
)
