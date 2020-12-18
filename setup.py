#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io
import re
import ast
from glob import glob
import os
from os.path import basename
from os.path import splitext
from setuptools import find_packages, setup
from distutils.extension import Extension

NAME = 'yass-algorithm'
DESCRIPTION = 'YASS: Yet Another Spike Sorter'
URL = 'https://github.com/paninski-lab/yass'
EMAIL = 'mitelutco@gmail.com'
AUTHOR = 'Peter Lee, Catalin Mitelut'
LICENSE = 'Apache'

# YASS dependencies
# NOTE: this are installed when running pip install yass-algorithm, however
# when building the documentation on readthedocs.io we do not install them
# and just mock them, if doc building breaks, make sure you update the
# autodoc_mock_imports list in conf.py
INSTALL_REQUIRES = [
    # these first two are only required for Python 2
    'pathlib2;python_version<"3"', 'funcsigs;python_version<"3"',
    # dependencies...
    'numpy', 'scipy', 'scikit-learn', 'pyyaml', 'python-dateutil', 'click',
    'tqdm', 'multiprocess', 'coloredlogs', 'cerberus', 
    # 'torch',
    # from experimental pipeline (nnet and clustering)
    # TODO: consider reducing the number of dependencies: parmap, matplotlib
    # and progressbar2 are not necessary
    'parmap', 'statsmodels', 'matplotlib', 'networkx', 'Cython', 'progressbar2',
    'h5py'
]

# this will be installed when doing `pip install yass-algorithm[tf]`
# or `pip install yass-algorithm[tf-gpu]
#EXTRAS_REQUIRE = {'tf': ['tensorflow'], 'tf-gpu': ['tensorflow-gpu']}

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

_version_re = re.compile(r'__version__\s+=\s+(.*)')

with open('src/yass/__init__.py', 'rb') as f:
    VERSION = str(ast.literal_eval(_version_re.search(
        f.read().decode('utf-8')).group(1)))

# Cython and numpy installation based on this:
# https://stackoverflow.com/a/42163080/709975


try:
    from Cython.setuptools import build_ext
except Exception:
    # If we couldn't import Cython, use the normal setuptools
    # and look for a pre-compiled .c file instead of a .pyx file
    from setuptools.command.build_ext import build_ext
    ext_modules = [Extension(name="diptest._diptest",
                             sources=["src/diptest/_dip.c",
                                      "src/diptest/_diptest.c"],
                             extra_compile_args=['-O3', '-std=c99'])]
else:
    # If we successfully imported Cython, look for a .pyx file
    ext_modules = [Extension(name="diptest._diptest",
                             sources=["src/diptest/_dip.c",
                                      "src/diptest/_diptest.pyx"],
                             extra_compile_args=['-O3', '-std=c99'])]


class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed
    """

    def run(self):

        # Import numpy here, only when headers are needed
        import numpy

        # Add numpy headers to include_dirs
        self.include_dirs.append(numpy.get_include())

        # Call original build_ext command
        build_ext.run(self)


setup(
    name=NAME,
    version=VERSION,
    license=LICENSE,
    description=DESCRIPTION,
    long_description=long_description,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    install_requires=INSTALL_REQUIRES,
    #extras_require=EXTRAS_REQUIRE,
    entry_points={
        'console_scripts': ['yass=yass.command_line:cli'],
    },
    download_url='{url}/archive/{version}.tar.gz'.format(url=URL,
                                                         version=VERSION),
    # diptest parameters
    cmdclass={'build_ext': CustomBuildExtCommand},
    ext_modules=ext_modules,
)
