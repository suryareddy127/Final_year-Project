from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="RAG Medical assistatnt",
    description="RAG Medical healthcare services assistant",
    version="0.1",
    author="Batch 04 - Final Project",
    packages=find_packages(),
    install_requires = requirements,
)