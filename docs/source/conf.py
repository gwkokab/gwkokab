# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from __future__ import annotations

import inspect
import operator
import os
import sys


sys.path.insert(0, os.path.abspath("../.."))


from gwkokab import __version__


project = "GWKokab"
copyright = "2024, Meesum Qazalbash, Muhammad Zeeshan, Richard O'Shaughnessy"
author = "Meesum Qazalbash, Muhammad Zeeshan, Richard O'Shaughnessy"
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_tabs.tabs",
    "sphinx_design",
    "nbsphinx",
    "myst_parser",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_search.extension",
    "sphinx.ext.linkcode",
    "sphinx_remove_toctrees",
    "sphinx_copybutton",
]

autodoc_inherit_docstrings = True
templates_path = ["_templates"]
source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".ipynb": "jupyter_notebook",
    ".md": "markdown",
}
nbsphinx_execute = "never"
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_css_files = ["style.css"]
html_theme = "sphinx_book_theme"
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.png"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "show_toc_level": 2,
    "repository_url": "https://github.com/gwkokab/gwkokab",
    "use_repository_button": True,  # add a "link to repository" button
    "navigation_with_keys": False,
    "use_download_button": True,
}


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
    ".DS_Store",
    "_build",
    "**.ipynb_checkpoints",
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "default"


autosummary_generate = True
napolean_use_rtype = False

# Remove auto-generated API docs from sidebars. They take too long to build.
remove_from_toctrees = ["_autosummary/*"]


# do not prepend module name to functions
add_module_names = False

autodoc_type_aliases = {
    "Iterable": "Iterable",
    "ArrayLike": "ArrayLike",
    "Numeric": "Numeric",
}


def linkcode_resolve(domain, info):
    import gwkokab

    if domain != "py":
        return None
    if not info["module"]:
        return None
    if not info["fullname"]:
        return None
    if info["module"].split(".")[0] != "gwkokab":
        return None
    try:
        mod = sys.modules.get(info["module"])
        obj = operator.attrgetter(info["fullname"])(mod)
        if isinstance(obj, property):
            obj = obj.fget
        while hasattr(obj, "__wrapped__"):  # decorated functions
            obj = obj.__wrapped__
        filename = inspect.getsourcefile(obj)
        source, linenum = inspect.getsourcelines(obj)
    except Exception:
        return None
    filename = os.path.relpath(filename, start=os.path.dirname(gwkokab.__file__))
    lines = f"#L{linenum}-L{linenum + len(source)}" if linenum else ""
    return f"https://github.com/gwkokab/gwkokab/blob/main/gwkokab/{filename}{lines}"


os.system("cp -r ../../examples .")


intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "typing_extensions": ("https://typing-extensions.readthedocs.io/en/latest/", None),
    "numpyro": ("https://num.pyro.ai/en/stable/", None),
    "jaxtyping": ("https://docs.kidger.site/jaxtyping/", None),
}
