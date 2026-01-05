"""
Setup script for compiling Cython extension module.
===================================================
Modules to be compiled:
- pyafv/cell_geom.pyx

---------------------------------------------------
Created by Wei Wang, 2025.
"""

from setuptools import setup, Extension
import numpy as np
# from Cython.Build import cythonize


ext = Extension(
    "pyafv.cell_geom",                # extension module name
    sources=["pyafv/cell_geom.cpp"],  # remove Cython dependency at build time by using the pre-generated C++ file
    extra_compile_args=[],
    extra_link_args=[],
    include_dirs=[np.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    language="c++",
)

setup(
    # name="pyafv",    # project name
    # ext_modules=cythonize([ext], language_level="3", build_dir="build"),
    # Remove Cython dependency at build time by generating .cpp file from .pyx beforehand
    # with (uv run)+`cython -3 --cplus pyafv/cell_geom.pyx -o pyafv/cell_geom.cpp`
    ext_modules=[ext]  # without cythonize
)
