from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("model/requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="next-gen-stock-prediction",
    version="2.0.0",
    author="Rajdeep Roy",
    author_email="rajdeep.roy.183@gmail.com",
    description="Next-Generation Stock Prediction with Advanced Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Rajdeep183/stock_pred",
    packages=find_packages(),
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
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
        ],
        "gpu": [
            "torch[cuda]>=1.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "stock-predict=cloud.functions.lib.model.stock_lstm:main",
        ],
    },
)