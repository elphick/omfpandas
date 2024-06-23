# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))

import omfpandas

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'omfpandas'
copyright = '2024, Greg Elphick'
author = 'Greg Elphick'
version = omfpandas.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.autosummary',  # to document the api
              'sphinx.ext.viewcode',  # to add view code links
              'sphinx.ext.coverage',
              'sphinx.ext.napoleon',  # for parsing numpy/google docstrings
              'sphinx_gallery.gen_gallery',  # to generate a gallery of examples
              'sphinx_autodoc_typehints',
              'myst_parser',  # for parsing md files
              'sphinx.ext.todo'
              ]

autosummary_generate = True
sphinx_gallery_conf = {
    'filename_pattern': r'\.py',
    'ignore_pattern': r'(__init__)\.py',
    'examples_dirs': '../../examples',
    'gallery_dirs': 'auto_examples',
    'nested_sections': False,
    'download_all_examples': True,
    'within_subsection_order': 'FileNameSortKey',
}

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# to widen the page...
html_css_files = ['custom.css']
