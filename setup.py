import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="bayesian-ntk",
    version="0.0.1",
    author="bobby-he",
    description="Code for Bayesian Deep Ensembles via the Neural Tangent Kernel",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bobby-he/bayesian-ntk",
    python_requires='>=3.6',
    packages=setuptools.find_packages(),
    # The following may be untrue and needs to be checked!
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
