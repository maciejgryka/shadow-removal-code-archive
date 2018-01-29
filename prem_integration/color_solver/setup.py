from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

ext_modules = [Extension(
    "color_solver",
    ["color_solver.pyx"],
    include_dirs=[numpy.get_include()],
)]

setup(
  name = 'color_solver',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)