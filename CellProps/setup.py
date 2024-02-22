#! /ur/bin/env python
import setuptools

DESCRIPTION = "Calculate image properties"
URL = "https://github.com/cwood1967"
LICENSE = 'MIT'
VERSION = '0.0.1'
PYTHON_REQUIRES = ">=3.7"

INSTALL_REQUIRES = [
    'numpy>=1.15',
    'scipy>=1.0',
    'matplotlib>=2.2',
]

if __name__ == "__main__":
    from setuptools import setup

    import sys
    if sys.version_info[:2] < (3, 6):
        raise RuntimeError("cellprops python >= 3.6.")

    setup(
        name='cellprops',
        author="Chris Wood",
        author_email="cjw@stowers.org",
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        python_requires=PYTHON_REQUIRES,
        install_requires=INSTALL_REQUIRES,
        packages=setuptools.find_packages(),
    )
