from setuptools import setup, find_packages


setup(
    name="linguify",
    version="0.1",
    packages=find_packages(),
    package_data={
        # Include any non-Python files that should be part of the package
        "linguify": ["*.md"],
    },
    install_requires=[

    ],
    description="Because conversations should resonate.",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/josephchay/linguify",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
