import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="bayesian-ntk",
    version="0.0.1",
    author="Bobby He",
    description="Code for Bayesian Deep Ensembles via the Neural Tangent Kernel",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.6',
    packages=setuptools.find_packages(),
    # The following may be untrue and needs to be checked!
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
