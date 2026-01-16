#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import re
import sys
import urllib.parse
from datetime import datetime
import re
from dataclasses import is_dataclass

sys.path.insert(0, os.path.abspath('_ext'))
sys.path.insert(0, os.path.abspath('docs'))

def get_version_from_file(version_file, full=True):
    with open(version_file, 'r') as f:
        version = f.read().strip()
        if full:
            return version
        match = re.search(r"(\d+)(\.\d+)+", version)
        return match.group(0)

# -- Project information -----------------------------------------------------
project = 'AMD Quark'

# Sphinx automatically adds the copyright to the footer of every page it generates
copyright = '2024, Advanced Micro Devices, Inc'
author = 'Advanced Micro Devices, Inc'

# The short X.Y version
version = get_version_from_file(os.path.join('..', '..', 'quark', 'version.txt'), full=False)
# The full version, including alpha/beta/rc tags
release = get_version_from_file(os.path.join('..', '..', 'quark', 'version.txt'), full=True)
# The short X.Y version
html_last_updated_fmt = datetime.now().strftime('%b %d, %Y')

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'autoapi.extension',
    'breathe',
    'myst_nb',
    'notfound.extension',
    'quark_version_substitution',
    'quark_jupyter_notebook_build',
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.githubpages',
    'sphinx.ext.graphviz',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
]

# Auto API settings
autoapi_dirs = ['../../quark']
autoapi_keep_files = True
autoapi_add_toctree_entry = False
autoapi_options = ["members", "show-module-summary"]
autoapi_ignore = ['*/quark/contrib/*']  # TODO: include contrib into documentation soon
                                        # TODO: https://github.com/readthedocs/sphinx-autoapi/issues/312 must use *subfolder* pattern

FACTORY_TYPES = {"typing.List": "[]", "list": "[]", "typing.Dict": "{}", "dict": "{}", "str": "''"}

def fix_signature(app, what, name, obj, options, signature, return_annotation):
    """
    Fixes a formatting bug in sphinx-autodoc with ``dataclasses.dataclass``'s ``default_factory`` reported in: https://github.com/sphinx-doc/sphinx/issues/10893 and https://github.com/sphinx-doc/sphinx/issues/12695.
    In case the signature contains "<" or ">", its rendering breaks with things like "~typing.Optional[~quark.torch.quantization.config.type.QuanitzationMode]" being rendered.
    This functions fixes the generated ``signature`` string and replaces the substring ``"<factory>"`` with the relevant default for each data type (``{}`` for dict, ``[]`` for list, ``""`` for string)

    TODO: remove once https://github.com/sphinx-doc/sphinx/issues/10893 and https://github.com/sphinx-doc/sphinx/issues/12695 are fixed.
    """
    if what == "class" and is_dataclass(obj):
        # Avoid splitting `Dict[str, str]`, for example.
        split_signature = re.split(r",(?![^\[\]]*\])", signature)

        split_signature[0] = split_signature[0][1:]  # Remove leading '('
        split_signature[-1] = split_signature[-1][:-1]  # Remove trailing ')'

        fixed_signature = "("
        for i, arg in enumerate(split_signature):
            if "<factory>" not in arg:
                fixed_signature += arg
            else:
                cur_idx = -1
                default = None

                for type in FACTORY_TYPES:
                    idx = arg.find(type)

                    if idx > -1 and (cur_idx == -1 or idx < cur_idx):
                        default = FACTORY_TYPES[type]
                        break

                if default is None:
                    raise RuntimeError(f"Unexpected data type for <factory>. Details:\n\twhat={what}, \n\tname={name}, \n\tobj={obj}, \n\toptions={options}, \n\tsignature={signature}, \n\treturn_annotation={return_annotation}")

                fixed_arg = arg.replace("<factory>", default)

                fixed_signature += fixed_arg

            if i < len(split_signature) - 1:
                fixed_signature += ","

        fixed_signature += ")"

        return fixed_signature, return_annotation

# TODO: Revisit when to ignore classes
# Ignore `WARNING: py:class reference target not found: torch.nn.Module`, etc.

nitpick_ignore_regex = [(r'py:class', r'.*')]

# TODO: Remove once https://github.com/sphinx-doc/sphinx/issues/4961 is addressed.
# Bypass `WARNING: more than one target found for cross-reference 'Config': quark.onnx.quantization.config.config.Config, quark.torch.pruning.config.Config, quark.torch.quantization.config.config.Config`, etc.
suppress_warnings = [
    'ref.python',
    'autoapi.python_import_resolution'
]

graphviz_output_format = 'svg'

# Prefix document path to section labels, otherwise autogenerated labels would look like 'heading'
# rather than 'path/to/file:heading'
autosectionlabel_prefix_document = True

# Breathe Configuration
breathe_projects = {
    "XRT":"../xml",
}

# Configuration for rst2pdf
pdf_documents = [('index', u'', u'', u'AMD, Inc.'),]

# Configure 'Edit on GitHub' extension
edit_on_github_project = '/amd/quark'
edit_on_github_branch = 'main/docs'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# Expand/Collapse functionality
def setup(app):
    app.add_css_file('custom.css')
    app.connect("autodoc-process-signature", fix_signature)

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['include', 'api_rst', '_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']
exclude_patterns.append('*autoapi/quark/index.rst')

nitpicky = True

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

primary_domain = 'c'
highlight_language = 'none'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'sphinx_book_theme'
html_theme = 'rocm_docs_theme'

# For 'rocm_docs_theme' them
html_context = {}
html_context["projects"] = {"quark": "https://quark.docs.amd.com"}
if "READTHEDOCS" in os.environ:
    html_context["READTHEDOCS"] = True

##html_theme_path = ["./_themes"]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#

# Add any theme-specific options here
# Add this part to expand the TOC
html_theme_options = {
    'collapse_navigation': False,  # Set to False to expand all sections
}

##html_logo = '_static/xilinx-header-logo.svg'
external_projects_current_project = "quark"
html_theme_options = {
    # "flavor": "rocm-docs-home",
    "flavor": "local",
    "repository_url": "https://github.com/amd/quark",
    "repository_provider": "github",
    "link_main_doc": False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
##html_static_path = ['_static']
##html_css_files = ['_static/custom.css']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
#html_sidebars = {
#    '**': [
#        'about.html',
#        'navigation.html',
#        'relations.html',
#        'searchbox.html',
#        'donate.html',
#    ]}

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'ProjectName'

# -- Options for LaTeX output ------------------------------------------------
latex_engine = 'pdflatex'
latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
     'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
     'pointsize': '12pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'quark.tex', 'Quark',
     'AMD', 'manual'),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'quark.tex', 'Quark',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'ENTER YOUR LIBRARY ID HERE. FOR EXAMPLE: xfopencv', 'ENTER YOUR LIBRARY PROJECT NAME HERE',
     author, 'AMD', 'One line description of project.',
     'Miscellaneous'),
]

# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']

# -- Options for rinoh ------------------------------------------

rinoh_documents = [dict(doc='index',        # top-level file (index.rst)
                        target='manual')]   # output file (manual.pdf)

# -- Notfound (404) extension settings

if "READTHEDOCS" in os.environ:
    components = urllib.parse.urlparse(os.environ["READTHEDOCS_CANONICAL_URL"])
    notfound_urls_prefix = components.path

# -- Extension configuration -------------------------------------------------
# At the bottom of conf.py
#def setup(app):
#    app.add_config_value('recommonmark_config', {
#            'url_resolver': lambda url: github_doc_root + url,
#            'auto_toc_tree_section': 'Contents',
#            }, True)
#    app.add_transform(AutoStructify)

if "READTHEDOCS" not in os.environ:

    ## myst_nb default settings

    # Custom formats for reading notebook; suffix -> reader
    # nb_custom_formats = {}

    # Notebook level metadata key for config overrides
    # nb_metadata_key = 'mystnb'

    # Cell level metadata key for config overrides
    # nb_cell_metadata_key = 'mystnb'

    # Mapping of kernel name regex to replacement kernel name(applied before execution)
    # nb_kernel_rgx_aliases = {}

    # Regex that matches permitted values of eval expressions
    # nb_eval_name_regex = '^[a-zA-Z_][a-zA-Z0-9_]*$'

    # Execution mode for notebooks
    # nb_execution_mode = 'auto'

    # Path to folder for caching notebooks (default: <outdir>)
    # nb_execution_cache_path = ''

    # Exclude (POSIX) glob patterns for notebooks
    # nb_execution_excludepatterns = ()

    # Execution timeout (seconds)
    nb_execution_timeout = -1

    # Use temporary folder for the execution current working directory
    # nb_execution_in_temp = False

    # Allow errors during execution
    # nb_execution_allow_errors = False

    # Raise an exception on failed execution, rather than emitting a warning
    nb_execution_raise_on_error = True

    # Print traceback to stderr on execution error
    nb_execution_show_tb = True

    # Merge stdout/stderr execution output streams
    nb_merge_streams = True

    # The entry point for the execution output render class (in group `myst_nb.output_renderer`)
    # nb_render_plugin = 'default'

    # Remove code cell source
    # nb_remove_code_source = False

    # Remove code cell outputs
    # nb_remove_code_outputs = False

    # Prompt to expand hidden code cell {content|source|outputs}
    # nb_code_prompt_show = 'Show code cell {type}'

    # Prompt to collapse hidden code cell {content|source|outputs}
    # nb_code_prompt_hide = 'Hide code cell {type}'

    # Number code cell source lines
    # nb_number_source_lines = False

    # Overrides for the base render priority of mime types: list of (builder name, mime type, priority)
    # nb_mime_priority_overrides = ()

    # Behaviour for stderr output
    # nb_output_stderr = 'show'

    # Pygments lexer applied to stdout/stderr and text/plain outputs
    # nb_render_text_lexer = 'myst-ansi'

    # Pygments lexer applied to error/traceback outputs
    # nb_render_error_lexer = 'ipythontb'

    # Options for image outputs (class|alt|height|width|scale|align)
    # nb_render_image_options = {}

    # Options for figure outputs (classes|name|caption|caption_before)
    # nb_render_figure_options = {}

    # The format to use for text/markdown rendering
    # nb_render_markdown_format = 'commonmark'

    # Javascript to be loaded on pages containing ipywidgets
    # nb_ipywidgets_js = {'https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js': {'integrity': 'sha256-Ae2Vz/4ePdIu6ZyI/5ZGsYnb+m0JlOmKPjt6XZ9JJkA=', 'crossorigin': 'anonymous'}, 'https://cdn.jsdelivr.net/npm/@jupyter-widgets/html-manager@1.0.6/dist/embed-amd.js': {'data-jupyter-widgets-cdn': 'https://cdn.jsdelivr.net/npm/', 'crossorigin': 'anonymous'}}
