# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from __future__ import annotations

import glob
import os
import shutil
import sys

import nbsphinx
import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath("../.."))

os.environ["SPHINX_BUILD"] = "1"

project = "GWKokab"
copyright = "2024, Meesum Qazalbash, Muhammad Zeeshan, Richard O'Shaughnessy"
author = "Meesum Qazalbash, Muhammad Zeeshan, Richard O'Shaughnessy"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "nbsphinx",
    "sphinxcontrib.jquery",
    "sphinx.ext.doctest",
    "sphinx.ext.imgconverter",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    # "sphinx_gallery.gen_gallery",
    "sphinx_search.extension",
]


templates_path = ["_templates"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_style = "css/gwk.css"
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# do not execute cells
nbsphinx_execute = "never"

# Don't add .txt suffix to source files:
html_sourcelink_suffix = ""

# The master toctree document.
master_doc = "index"

language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = [
    ".ipynb_checkpoints",
    "tutorials/logistic_regression.ipynb",
    "examples/*ipynb",
    "examples/*py",
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"


# do not prepend module name to functions
add_module_names = False

autodoc_type_aliases = {
    "Iterable": "Iterable",
    "ArrayLike": "ArrayLike",
    "Numeric": "Numeric",
}
