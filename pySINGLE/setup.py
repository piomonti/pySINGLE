#from distutils.core import setup
#from distutils.extension import Extension
#from Cython.Distutils import build_ext
#import numpy

#setup(
    #cmdclass = {'build_ext': build_ext},
    #ext_modules = [Extension("Z_shooting", ["Z_shooting.c"],)],
    #include_dirs=[numpy.get_include(),'.', ],
#)


from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

#extension = [Extension("Z_shooting", ["Z_shooting.c"],),]

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("FastFused_01", ["FastFused_01.pyx"], include_dirs=[numpy.get_include()])]
)

#setup(
    #cmdclass = {'build_ext': build_ext},
    #ext_modules = cythonize("FastFused_01.pyx"),
    #include_dirs=[numpy.get_include(),'.', ],
#)
