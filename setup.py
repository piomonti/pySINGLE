#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

cythonize("pySINGLE/FastFused_01.pyx")

setup(name='pySINGLE',
    packages = ['pySINGLE'],
    version='0.5',
    description = 'Python implementation of SINGLE algorithm',
    author = 'Ricardo Pio Monti',
    author_email = 'ricardo.monti08@gmail.com',
    url = 'https://github.com/piomonti/pySINGLE',
    download_url = 'https://github.com/piomonti/pySINGLE/tarball/0.1',
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("FastFused_01", ['pySINGLE/FastFused_01.c'])],
    include_dirs=[numpy.get_include(),'.', ]
)
