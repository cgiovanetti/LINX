# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../linx'))
sys.path.append('..')

project = 'LINX'
copyright = '2024, Cara Giovanetti, Mariangela Lisanti, Hongwan Liu, Siddharth Mishra-Sharma, and Joshua T. Ruderman'
author = 'Cara Giovanetti, Mariangela Lisanti, Hongwan Liu, Siddharth Mishra-Sharma, and Joshua T. Ruderman'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_default_flags = ['members', 'undoc-members', 'special-members']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'bizstyle'
html_theme_options = {
    'sidebarwidth': 350
}
html_static_path = ['_static']
