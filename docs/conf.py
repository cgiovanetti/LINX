# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = 'LINX'
copyright = '2026, Cara Giovanetti, Mariangela Lisanti, Hongwan Liu, Siddharth Mishra-Sharma, and Joshua T. Ruderman'
author = 'Cara Giovanetti, Mariangela Lisanti, Hongwan Liu, Siddharth Mishra-Sharma, and Joshua T. Ruderman'

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../linx'))

import re
from sphinx.util import logging

logger = logging.getLogger(__name__)

def format_method_summaries(app, what, name, obj, options, lines):
    """
    Process docstrings to ensure NumPy-style sections are properly formatted.
    Handles sections with hyphen underlining (Parameters, Returns, etc.).
    """
    out = []
    in_prose_section = False
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if next line is a hyphen underline (NumPy convention)
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            if re.match(r'^\s*-+\s*$', next_line) and line.strip():
                # Check if this is a prose section (Parameters, Returns, etc.)
                in_prose_section = bool(re.search(r'(Parameters|Returns|Notes|Examples|Attributes):', line, re.IGNORECASE))
                
                # Output the header line with bold formatting
                out.append(f"**{line.strip()}**")
                # Skip the hyphen line
                i += 1
                # Add blank line for proper reST parsing
                out.append("")
                i += 1
                continue
        
        # Blank line: close any section
        if line.strip() == "":
            in_prose_section = False
        
        # Default passthrough
        out.append(line)
        i += 1
    
    # Mutate the list in-place
    lines[:] = out


def setup(app):
    app.connect('autodoc-process-docstring', format_method_summaries)


# -- General configuration ---------------------------------------------------

templates_path = ['_templates']
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax'
]

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_default_options = {
    'special-members': '__call__',
    'members': True,
    'show-inheritance': True
}

# Napoleon settings for NumPy-style docstrings
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
