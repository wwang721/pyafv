# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

from datetime import datetime, timezone
import importlib.metadata
import os.path
import sys


sys.path.insert(0, os.path.abspath("."))  # adds docs/ to sys.path

# -- Project information -----------------------------------------------------

project = "PyAFV"
author = "Wei Wang"
copyright = f"{datetime.now(timezone.utc).year} {author}"
release = importlib.metadata.version("pyafv")

# -- General configuration ---------------------------------------------------
# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    # added extensions
    'sphinxcontrib.bibtex',
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    # custom extensions in docs/_ext
    "_ext.subst_release",
]

bibtex_bibfiles = ['main.bib']

intersphinx_mapping = {
    "rtd": ("https://docs.readthedocs.io/en/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for EPUB output
epub_show_urls = "footnote"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    'version_selector': True
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]

# Create table of contents entries for domain objects (e.g. functions, classes, attributes, etc.).
toc_object_entries = False

autodoc_type_aliases = {
    # 'np.ndarray': 'numpy.ndarray',
    # 'Axes': 'matplotlib.axes.Axes',
    'Voronoi': 'scipy.spatial.Voronoi',
    # Somehow numpy/matplotlib not working,
    # only scipy.spatial.Voronoi works here.
}

napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Make Sphinx show type hints (from function signatures)
autodoc_typehints = "description"  # or "signature"
autodoc_typehints_format = "short"

# Suppress the module name of the python reference if it can be resolved.
python_use_unqualified_type_names = True
