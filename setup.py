"""
Setup script for compiling Cython extension module.
===================================================
Modules to be compiled:
- cell_geom.pyx

---------------------------------------------------
Created by Wei Wang, 2025.
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np


ext = Extension(
    "afv.cell_geom",
    sources=["afv/cell_geom.pyx"],
    extra_compile_args=[],
    extra_link_args=[],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    language="c++",
)

setup(
    name="cell_geom",
    ext_modules=cythonize([ext], language_level="3", build_dir="build"),
    include_dirs=[np.get_include()],
)
