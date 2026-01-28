# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import datetime
import os
import sys


sys.path.insert(0, os.path.abspath("../.."))
sys.path.append(os.path.abspath("./_pygments"))


project = "GWKokab"
copyright = f"2023-{datetime.date.today().year}, Meesum Qazalbash, Muhammad Zeeshan, Richard O'Shaughnessy"
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
    "autoapi.extension",
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_togglebutton",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
]

bibtex_encoding = "latin"
bibtex_default_style = "unsrt"
bibtex_bibfiles = ["references.bib"]

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
autoapi_python_class_content = "both"
autoapi_member_order = "groupwise"
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


html_static_path = ["_static"]

html_css_files = ["style.css"]
html_theme_options = {
    "sidebar_hide_name": True,
    "light_logo": "noBgBlack.png",
    "dark_logo": "noBgWhite.png",
}
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
            f"https://github.com/gwkokab/gwkokab/blob/HEAD/src/gwkokab/{fn}{linespec}"
        )
    else:
        return f"https://github.com/gwkokab/gwkokab/blob/v{gwkokab.__version__}/src/gwkokab/{fn}{linespec}"


intersphinx_mapping = {
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "jaxtyping": ("https://docs.kidger.site/jaxtyping/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "numpyro": ("https://num.pyro.ai/en/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "chex": ("https://chex.readthedocs.io/en/latest/", None),
}


# source: https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html#event-autoapi-skip-member
def skip_util_classes(app, what, name: str, obj, skip, options):
    if what == "module" and "._" in name:  # skip private modules
        skip = True
        return skip
    if what == "method":  # skip private modules
        if "tree_flatten" in name or "tree_unflatten" in name:
            skip = True
            return skip
    if what == "attribute" and (
        "pytree_data_fields" in name
        or "pytree_aux_fields" in name
        or "arg_constraints" in name
        or "reparametrized_params" in name
    ):
        skip = True
        return skip
    if what == "module" and "cli_gwkokab" in name:
        skip = True
        return skip
    if (
        what == "module"
        and name.startswith("kokab")
        and (
            "sage" in name
            or "genie" in name
            or "common" in name
            or "ppd" in name
            or "monk" in name
        )
    ):
        skip = True
        return skip
    return skip


def setup(sphinx):
    sphinx.connect("autoapi-skip-member", skip_util_classes)
