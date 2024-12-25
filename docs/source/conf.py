# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import os
import sys


sys.path.insert(0, os.path.abspath("../.."))
sys.path.append(os.path.abspath("./_pygments"))


project = "GWKokab"
copyright = "2024, Meesum Qazalbash, Muhammad Zeeshan, Richard O'Shaughnessy"
author = "Meesum Qazalbash, Muhammad Zeeshan, Richard O'Shaughnessy"


try:
    from gwkokab import __version__

    release = __version__
except ImportError:
    try:
        from importlib.metadata import version as _version

        release = _version("gwkokab")
    except ImportError:
        release = "0.0.0+unknown"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "sphinx_copybutton",
    "autoapi.extension",
    # "myst_parser",
    "sphinx_design",
]


nb_execution_mode = "off"
myst_heading_anchors = 4
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

# sphinx-autoapi
autoapi_dirs = ["../../src"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"

html_theme_options = {
    "sidebar_hide_name": True,
}

html_static_path = ["_static"]

html_css_files = ["style.css"]
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.png"

pygments_style = "_pygments_light.MarianaLight"
pygments_dark_style = "_pygments_dark.MarianaDark"


def linkcode_resolve(domain, info):
    """Determine the URL corresponding to Python object."""
    import inspect

    import gwkokab

    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        fn = inspect.getsourcefile(inspect.unwrap(obj))
    except TypeError:
        fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        lineno = None

    if lineno:
        linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
    else:
        linespec = ""

    fn = os.path.relpath(fn, start=os.path.dirname(gwkokab.__file__))

    if "+" in gwkokab.__version__:
        return (
            f"https://github.com/gwkokab/gwkokab/blob/"
            f"HEAD/src/gwkokab/{fn}{linespec}"
        )
    else:
        return (
            f"https://github.com/gwkokab/gwkokab/blob/"
            f"v{gwkokab.__version__}/src/gwkokab/{fn}{linespec}"
        )


intersphinx_mapping = {
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "jaxtyping": ("https://docs.kidger.site/jaxtyping/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "numpyro": ("https://num.pyro.ai/en/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "typing_extensions": ("https://typing-extensions.readthedocs.io/en/latest/", None),
    "chex": ("https://chex.readthedocs.io/en/latest/", None),
}


# source: https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html#event-autoapi-skip-member
def skip_util_classes(app, what, name, obj, skip, options):
    if what == "module" and "._" in name:  # skip private modules
        skip = True
    return skip


def setup(sphinx):
    sphinx.connect("autoapi-skip-member", skip_util_classes)
