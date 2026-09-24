MINCE: Monte-Carlo Informed N-sizing for Compact Evaluation
===========================================================

.. note::

   MINCE is a community contribution in Quark's ``contrib`` area. It is maintained by its author, `@devledas <https://github.com/devledas>`_.
   For questions, bugs, or feedback, please open a GitHub issue and tag
   ``@devledas``. See :doc:`/intro_contrib` for the ``contrib`` policies.

MINCE (Monte-Carlo Informed N-sizing for Compact Evaluation) cuts LLM
benchmark evaluation time by evaluating a small,
frozen subset of items instead of the full benchmark, while keeping the subset
score close to the full-benchmark score.

Given a bf16 model's per-item evaluation logs, MINCE:

#. **sizes** a representative subset (``n*``) via a Monte-Carlo drift sweep,
#. **freezes** that subset into a reproducible, ID-based artifact plus an
   lm-eval ``--samples`` map, and
#. lets you **reuse** the frozen subset to evaluate downstream model variants
   within a bounded, quantified accuracy drift.

For the method and experiments, see the MINCE paper:
`arxiv.org/abs/2606.22826 <https://arxiv.org/pdf/2606.22826>`_.

Package and examples
--------------------

The importable core lives in the ``quark.contrib.mince`` package
(``config``, ``data_loader``, ``mince_metrics``, ``montecarlo``, ``selection``,
``subset``), with unit tests under ``quark/contrib/mince/test``. The runnable
CLIs (``size.py``, ``freeze.py``, ``validate.py``, ``extract_inputs.py``) live
under ``examples/contrib/mince``.

The end-to-end walkthrough — generate bf16 logs, size ``n*``, freeze the subset,
score it with lm-eval, and validate the drift — is below. We also share an
example that reuses a frozen subset to evaluate a Quark-quantized model. We provide a guide
to adding benchmarks beyond those that ship with MINCE. We also provide a guide for calibrating
``n*`` on several bf16 models instead of one.

.. toctree::
   :maxdepth: 2

   example_quark_torch_mince
   example_quark_torch_mince_quantized
   adding_benchmarks
   multi_model_calibration
