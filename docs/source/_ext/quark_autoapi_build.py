#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import os
from sphinx.util import logging

logger = logging.getLogger(__name__)

def update_autoapi_toc_placeholder(app, docname, source):
    """
    Replace @quark_autoapi_toc_placeholder@ with actual autoapi toc when QUARK_SKIP_DOC_AUTOAPI is not set on env var
    """

    quark_autoapi_toc = """
.. toctree::
   :hidden:
   :caption: APIs
   :maxdepth: 1

   PyTorch APIs <autoapi/pytorch_apis>
   ONNX APIs <autoapi/onnx_apis>
"""

    generate_autoapi_docs = "QUARK_SKIP_DOC_AUTOAPI" not in os.environ or os.environ["QUARK_SKIP_DOC_AUTOAPI"].lower() in ("0", "false", "off")
    if not generate_autoapi_docs:
        quark_autoapi_toc = ""

    if '@quark_autoapi_toc_placeholder@' in source[0]:  # Only process documents containing @quark_autoapi_toc_placeholder@
        source[0] = source[0].replace('@quark_autoapi_toc_placeholder@', quark_autoapi_toc)
        if len(quark_autoapi_toc) > 0:
            logger.info(f'Replaced @quark_autoapi_toc_placeholder@ with AutoAPI table of Content defined at {__file__} in {docname}.rst')
        else:
            logger.info(f'Removed @quark_autoapi_toc_placeholder@ from {docname}.rst')

def setup(app):
    """
    Setup Sphinx extension
    """
    app.connect('source-read', update_autoapi_toc_placeholder)
    return {
        'version': '1.0',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
