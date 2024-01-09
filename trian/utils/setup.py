from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(ext_modules=cythonize(Extension('project_depth_nn_cython_pkg',
                                      sources=['project_depth_nn_cython.pyx'],
                                      language='c',
                                      include_dirs=[numpy.get_include()],
                                      library_dirs=[],
                                      libraries=[],
                                      extra_compile_args=[],
                                      extra_link_args=[], ),
                            language_level=3,
                            annotate=True))
