import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text(encoding="utf-8")

setup(
    name="hawkesbook",
    version="0.1.0",
    description="Hawkes process methods for inference, simulation, and related calculations",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Pat-Laub/hawkesbook",
    author="Patrick Laub",
    author_email="patrick.laub@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Intended Audience :: Science/Research",
    ],
    packages=["hawkesbook"],
    include_package_data=True,
    install_requires=["numba", "numpy", "scipy", "tqdm"],
)
