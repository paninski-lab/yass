#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io
import re
import ast
from glob import glob
import os
import sys
from os.path import basename
from os.path import splitext
from shutil import rmtree

from setuptools import find_packages, Command, setup

# yass was taken...
NAME = 'yass-algorithm'
DESCRIPTION = 'YASS: Yet Another Spike Sorter'
URL = 'https://github.com/paninski-lab/yass'
EMAIL = 'fkq8@blancas.io'
AUTHOR = 'Peter Lee, Eduardo Blancas'
LICENSE = 'Apache'

INSTALL_REQUIRES = [
    'numpy', 'scipy', 'scikit-learn', 'tensorflow', 'pyyaml',
    'python-dateutil', 'click',
    # FIXME: we should remove this
    'progressbar2',
    # FIXME: stability metric depends on this, remove...
    'tqdm'
]

# pass an empty INSTALL_REQUIRES if building the docs, to avoid breaking the
# build, modules are mocked in conf.py
INSTALL_REQUIRES = [] if os.environ.get('READTHEDOCS') else INSTALL_REQUIRES

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

_version_re = re.compile(r'__version__\s+=\s+(.*)')

with open('src/yass/__init__.py', 'rb') as f:
    VERSION = str(ast.literal_eval(_version_re.search(
        f.read().decode('utf-8')).group(1)))


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds...')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution...')
        os.system('{0} setup.py sdist bdist_wheel '
                  '--universal'.format(sys.executable))

        self.status('Uploading the package to PyPi via Twine...')
        os.system('twine upload dist/*')

        sys.exit()


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
    entry_points={
        'console_scripts': ['yass=yass.command_line:cli'],
    },
    download_url='{url}/archive/{version}.tar.gz'.format(url=URL,
                                                         version=VERSION),
)
