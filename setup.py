from setuptools import setup, find_packages

setup(
    name="breakout_thesis",
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "streamlit",
        "pandas",
        "yfinance",
        "plotly",
        "numpy",
        "matplotlib",
    ],
)