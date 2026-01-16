# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

# -- Project information -----------------------------------------------------

project = "RyzenAI ONNX Utils"
copyright = "2024 Advanced Micro Devices, Inc."
author = "Advanced Micro Devices, Inc."

# override this value to build different versions
version = "main"

release = f"v{version}" if version != "main" and version != "dev" else version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinxcontrib.jquery",
    # adds argparse directive to parse CLIs
    "sphinxarg.ext",
    "sphinx.ext.autodoc",
    # automatically labels headings
    "sphinx.ext.autosectionlabel",
    # used to define templatized links (see config below)
    "sphinx.ext.extlinks",
    # adds .nojekyll to the generated HTML for GitHub
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_issues",
    "sphinx_tabs.tabs",
    # adds tooltips
    "sphinx_tippy",
    # add emoji
    "sphinxemoji.sphinxemoji",
    "sphinx_favicon",
    "sphinx_rtd_theme",
    "sphinx_toolbox.collapse",
]

# sphinx.ext.autodoc configuration
autodoc_default_options = {
    "members": True,
    "special-members": "__init__",
}

tippy_add_class = "has-tippy"
tippy_skip_urls = [
    # skip all URLs except those pointing to the glossary
    r"^((?!terminology\.html).)*$"
]
tippy_enable_wikitips = False
tippy_enable_doitips = False

sphinxemoji_style = "twemoji"


def hide_private_module(app, what, name: str, obj, options, signature, return_annotation):
    if signature is not None:
        new_signature = signature.replace("ryzenai_onnx_utils._ryzenai_onnx_utils", "ryzenai_onnx_utils")
    else:
        new_signature = signature

    if return_annotation is not None:
        new_return = return_annotation.replace("ryzenai_onnx_utils._ryzenai_onnx_utils", "ryzenai_onnx_utils")
    else:
        new_return = return_annotation

    return (new_signature, new_return)


# sphinx.ext.autosectionlabel configuration
# prefix all generated labels with the document
autosectionlabel_prefix_document = True
# only auto-label top-level headings to prevent duplication when the same title
# is used in different sections
autosectionlabel_maxdepth = 3

# sphinx.ext.extlinks configuration. syntax is key: (url, caption). The key should not have underscores.
tree_path = f"https://gitenterprise.xilinx.com/varunsh/onnx_utils/tree/{release}/%s"
blob_path = f"https://gitenterprise.xilinx.com/varunsh/onnx_utils/blob/{release}/%s"
extlinks = {
    "onnxUtilsTree": (tree_path, "%s"),
    "onnxutilsBlob": (blob_path, "%s"),
}

# sphinx-issues configuration
issues_default_group_project = "varunsh/onnx_utils"

# strip leading $ from bash code blocks
copybutton_prompt_text = "$ "
copybutton_here_doc_delimiter = "EOF"
# selecting the literal block doesn't work to show the copy button correctly
# copybutton_selector = ":is(div.highlight pre, pre.literal-block)"

# raise a warning if a cross-reference cannot be found
nitpicky = True

# number all figures with captions
numfig = True


# Configure 'Edit on GitHub' extension
edit_on_github_project = "varunsh/onnx_utils"
edit_on_github_branch = f"{release}/docs"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**/build/**",
    "**/uploads/**",
]

# ignore duplicate label warnings
suppress_warnings = ["autosectionlabel.*"]


# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"

html_last_updated_fmt = "%B %d, %Y"

favicons = [
    {"rel": "apple-touch-icon", "sizes": "180x180", "href": "apple-touch-icon.png"},
    {"rel": "icon", "type": "image/png", "sizes": "32x32", "href": "favicon-32x32.png"},
    {"rel": "icon", "type": "image/png", "sizes": "16x16", "href": "favicon-16x16.png"},
    {"rel": "manifest", "href": "site.webmanifest"},
    {"rel": "mask-icon", "href": "safari-pinned-tab.svg", "color": "#5bbad5"},
    {"name": "msapplication-TileColor", "content": "#ed1c24"},
    {"name": "theme-color", "content": "#ed1c24"},
]


def setup(app) -> None:
    app.connect("autodoc-process-signature", hide_private_module)
