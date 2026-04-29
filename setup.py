from setuptools import setup, find_packages

setup(
    name="whitebox",
    version="0.1.0",
    description="Python SDK for the WhiteBox AI decision observability API",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Spaceman Tech",
    author_email="hello@spacemantech.ai",
    url="https://github.com/spacemantech/whitebox-python",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "requests>=2.28",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
