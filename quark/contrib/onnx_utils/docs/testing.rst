..
    Copyright (c) 2024 Advanced Micro Devices, Inc.

Testing
=======

The repository uses `pytest <https://docs.pytest.org/en/stable/>`__ to act as the test runner.
To run tests:

.. code-block:: console

    pytest tests <other arguments>

Some handy built-in pytest options are:

* ``-s``: print outputs from tests
* ``-ra``: print a summary after the tests which shows why a test was skipped

In addition to the regular pytest arguments you could specify, there are some custom options that some tests require to run:

* ``--dd-root``: path to the DynamicDispatch source directory
* ``--dll-path``: path to the custom ops DLL
