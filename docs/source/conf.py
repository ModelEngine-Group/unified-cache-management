# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Unified Cache Manager"
copyright = "2025, Unified Cache Manager Team"
author = "Unified Cache Manager Team"
release = ""

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
    "sphinxcontrib.mermaid",
]

myst_enable_extensions = ["colon_fence", "substitution"]
# Treat ```mermaid fenced blocks as mermaid directives. This way the same
# source renders both on GitHub (native mermaid support) and on Sphinx /
# ReadTheDocs (via sphinxcontrib-mermaid).
myst_fence_as_directive = ["mermaid"]

# templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = project
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["css/logo.css"]
html_theme_options = {
    "path_to_docs": "docs/source",
    "repository_url": "https://github.com/ModelEngine-Group/unified-cache-management",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "logo": {
        "image_light": "logos/UCM-light.png",
        "image_dark": "logos/UCM-dark.png",
        "alt_text": "UCM",
    },
}

# language = 'zh_CN'
