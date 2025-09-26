#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import os

from sphinx.util import logging

logger = logging.getLogger(__name__)


def update_jupyter_notebook_toc_placeholder(app, docname, source):
    """
    Replace THE *.ipynb references by *.rst from @quark_jupyter_notebook_toc_placeholder@ toc when QUARK_SPHINX_BUILD_SKIP_TUTORIALS is set on env var

    This assumes that the all *.ipynb files were converted into *.rst files using nbconvert with something like:

    ```bash
    conda install -c conda-force pandoc
    pip install nbconvert
    jupyter nbconvert --to rst your_notebook.ipynb
    ```
    """

    jupyter_notebook_index_rst = os.path.join("source", "jupyter_notebook_index.rst_")
    if "READTHEDOCS" in os.environ:
        READTHEDOCS_REPOSITORY_PATH = os.environ.get("READTHEDOCS_REPOSITORY_PATH")
        jupyter_notebook_index_rst = os.path.join(
            READTHEDOCS_REPOSITORY_PATH, "docs", "source", "jupyter_notebook_index.rst_"
        )
    jupyter_notebook_toc_placeholder = "@quark_jupyter_notebook_toc_placeholder@"
    with open(jupyter_notebook_index_rst) as f:
        quark_jupyter_notebook_toc = f.read().strip()

    generate_jupyter_notebook_docs = "QUARK_SPHINX_BUILD_SKIP_TUTORIALS" not in os.environ or os.environ[
        "QUARK_SPHINX_BUILD_SKIP_TUTORIALS"
    ].lower() in ("0", "false", "off")
    if not generate_jupyter_notebook_docs:
        quark_jupyter_notebook_toc = quark_jupyter_notebook_toc.replace(".ipynb", ".rst")

    if (
        jupyter_notebook_toc_placeholder in source[0]
    ):  # Only process documents containing `jupyter_notebook_toc_placeholder`
        source[0] = source[0].replace(jupyter_notebook_toc_placeholder, quark_jupyter_notebook_toc)
        if ".rst" in quark_jupyter_notebook_toc:
            logger.info(
                f"Replaced {jupyter_notebook_toc_placeholder} with ReStructuredText tutorials at jupyter_notebook_index_rst and indexed by {docname}.rst"
            )
        else:
            logger.info(
                f"Replaced {jupyter_notebook_toc_placeholder} with Jupyter Notebook tutorials at jupyter_notebook_index_rst and indexed by {docname}.rst"
            )


def setup(app):
    """
    Setup Sphinx extension
    """
    app.connect("source-read", update_jupyter_notebook_toc_placeholder)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
