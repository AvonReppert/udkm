"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os     import path
import re

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='udkm',
    version='0.0.1',
    packages=['udkm'],
    url='https://github.com/AleksUDKM/udkm.git',  # Optional
    install_requires=['numpy', 'matplotlib', 'os',],  # Optional
    license='',
    author='Alexander von Reppert',
    author_email='reppert@uni-potsdam.de',
    description='Collection of frequently used functions in udkm group',  # Required
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional (see note above)
)