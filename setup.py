"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='udkm',
    version='0.1.0',
    packages=find_packages(),
    url='https://github.com/AleksUDKM/udkm',  # Optional
    install_requires=['numpy', 'matplotlib', 'lmfit', 'pandas', 'dill'],  # Optional
    extras_require={
        'documentation': ['sphinx', 'nbsphinx', 'sphinxcontrib-napoleon'],
    },
    license='',
    author='Alexander von Reppert',
    author_email='reppert@uni-potsdam.de',
    description='Collection of frequently used functions in udkm group',  # Required
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional (see note above)
    python_requires='>=3.5',
)
