# Quark documentation

Quark (new quantizer) documentation.

## How to build document

.. code-block:: sh

    sh install_requirements.sh
    sh build.sh

Access the HTML documentation at ``quark/_docs_build/html/index.html``.

.. raw:: html

   <!-- 
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->

## Collapsible sections

Collapsible sections can be added with:

    ```text
    .. container:: toggle

        .. container:: header

            //

        .. code-block:: python

            print("this code block will be collapsed by default")
    ```
