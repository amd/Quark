#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from sphinx.util import logging

logger = logging.getLogger(__name__)

def update_release_placeholder(app, docname, source):
    """
    Replace all occurrences of @version@ with actual version number
    """
    release_version = app.config.release
    if '@version@' in source[0]:  # Only process documents containing @version@
        source[0] = source[0].replace('@version@', release_version)
        logger.info(f'Replaced @version@ placeholder with Quark version {release_version} in {docname}.rst')

def setup(app):
    """
    Setup Sphinx extension
    """
    app.connect('source-read', update_release_placeholder)
    return {
        'version': '1.0',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
