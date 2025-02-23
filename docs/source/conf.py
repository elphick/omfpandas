# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

import numpy as np

import omfpandas
import pyvista

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'omfpandas'
copyright = '2024, Greg Elphick'
author = 'Greg Elphick'
version = omfpandas.__version__

# -- pyvista configuration ---------------------------------------------------

# Manage errors
pyvista.set_error_output_file("errors.txt")
# Ensure that offscreen rendering is used for docs generation
pyvista.OFF_SCREEN = True  # Not necessary - simply an insurance policy
pyvista.BUILDING_GALLERY = True  # necessary when building the sphinx gallery
# Preferred plotting style for documentation
pyvista.set_plot_theme("document")
pyvista.global_theme.window_size = np.array([1024, 768]) * 2

image_scrapers = ("pyvista", "matplotlib")


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
    "image_scrapers": image_scrapers,
}

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# to widen the page...
html_css_files = ['custom.css']
