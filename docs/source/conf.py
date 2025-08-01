# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Unified Cache Manager'
copyright = '2025, Unified Cache Manager Team'
author = 'Unified Cache Manager Team'
release = ''

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Copy from https://github.com/vllm-project/vllm/blob/main/docs/source/conf.py
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "myst_parser",
    "sphinxarg.ext",
    "sphinx_design",
    "sphinx_togglebutton",
    "sphinx_substitution_extensions",
]

myst_enable_extensions = ["colon_fence", "substitution"]

# templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = project
html_theme = 'sphinx_book_theme'
html_logo = 'logos/UCM.png'
html_theme_options = {
    'path_to_docs': 'docs/source',
    'repository_url': 'https://github.com/ModelEngine-Group/unified-cache-management',
    'use_repository_button': True,
    'use_edit_page_button': True,
}
# html_static_path = ['_static']

# language = 'zh_CN'
