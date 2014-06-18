#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

cythonize("pySINGLE/Z_shooting.pyx")

setup(name='pySINGLE',
    packages = ['pySINGLE'],
    version='0.5',
    description = 'Python implementation of SINGLE algorithm',
    author = 'Ricardo Pio Monti',
    author_email = 'ricardo.monti08@gmail.com',
    url = 'https://github.com/piomonti/pySINGLE',
    download_url = 'https://github.com/piomonti/pySINGLE/tarball/0.1',
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("Z_shooting", ['pySINGLE/Z_shooting.c'])],
    include_dirs=[numpy.get_include(),'.', ]
)