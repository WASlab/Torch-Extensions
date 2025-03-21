from setuptools import setup, find_packages

setup(
    name="torch-extensions",
    author="Waslab",
    author_email="Gradiorum@gmail.com",
    description="A collection of useful PyTorch extensions",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Waslab/Torch-Extensions",
    packages=find_packages(exclude=["tests*"]),
    install_requires=["torch"],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
