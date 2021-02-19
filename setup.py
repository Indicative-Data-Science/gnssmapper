import setuptools

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gnssmapper", 
    version="0.0.1",
    author="Terry Lines",
    author_email="terence.lines@glasgow.ac.uk",
    description="A package for generating 3D maps from gnss data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TerryLines/gnssmapper",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)