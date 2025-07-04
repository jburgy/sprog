# ruff: noqa: A001, D100, INP001

import sys
from pathlib import Path

sys.path.insert(0, str(Path("..", "..", "src").resolve()))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "sprog"
copyright = "2024, Jan Burgy"
author = "Jan Burgy"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_material"
html_static_path = ["_static"]

napoleon_google_docstring = True
napoleon_attr_annotations = True

intersphinx_mapping = {
    "python": ("http://docs.python.org/3", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/dev", None),
    "scipy": ("http://docs.scipy.org/doc/scipy/reference", None),
}

autosummary_generate = True
