# setup.py
from setuptools import setup, find_packages

setup(
    name="italian-political-compass",
    version="0.1.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch",
        "transformers",
    ],
    author="financialsupport",
    author_email="marvin994@gmail.com",
    description="A political compass tool using AI",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/political-compass",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
