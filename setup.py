import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="activeSVC",
    version="0.0.1",
    author="Xiaoqiao Chen",
    author_email="xqchen@caltech.edu",
    description="Active feature selection method with support vector classifier.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xqchen/activeSVC",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=2.7',
)