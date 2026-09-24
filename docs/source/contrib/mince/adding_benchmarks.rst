Adding a New Benchmark to MINCE
===============================

Adding a benchmark takes three code changes — a config entry, a loader, and a
``--samples`` mapping — plus a test fixture. Everything else in the pipeline
reads the config rather than hard-coding benchmark names.

This page uses GSM8K as its worked example because GSM8K ships in the repo, so
every snippet below can be checked against the real ``BENCHMARKS["gsm8k"]`` entry
in ``config.py`` and the real ``_load_gsm8k`` in ``data_loader.py``.

Step 1 — Read the task definition
---------------------------------

Most of what you need is already declared in the task's YAML in lm-eval-harness.
Read the file on the web under `lm_eval/tasks
<https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks>`_,
pinning the URL to the tag matching your installed version rather than browsing
``main`` — task definitions change between releases:

.. code-block:: bash

   python -c "import lm_eval; print(lm_eval.__version__)"
   # 0.4.12 -> .../lm-evaluation-harness/tree/v0.4.12/lm_eval/tasks

The YAML is not always named after the task. CommonsenseQA's lives at
``tasks/commonsense_qa/default.yaml``, and task YAMLs also inherit: ARC's
``arc_challenge.yaml`` is three lines that ``include: arc_easy.yaml``, with
``metric_list`` in the parent. Follow ``include:`` before concluding a field is
missing.

Three things come out of that file:

``metric_list[].metric``
  The metric names, which become ``metric_names``. GSM8K declares
  ``exact_match``; ARC declares ``acc`` and ``acc_norm``.

``filter_list[].name``
  If present, the task writes one row per filter and you pick one with
  ``sample_filter`` — GSM8K declares ``strict-match`` and ``flexible-extract``.
  If absent, it writes one row per item and you need no filter.

``aggregation`` on each metric
  ``mean`` on every metric means MINCE's existing scorer already handles the
  benchmark and you write no metrics code. An ``!function`` aggregation means you
  do; see `Metrics that are not a per-item mean`_.

Step 2 — Look at one sample row
-------------------------------

The YAML names the metrics but not the JSONL keys your loader has to read, and
those differ per task: the question text is under ``question`` for GSM8K and
CommonsenseQA, but ``goal`` for PIQA. Generate three rows with a tiny model to
see the real shape — no results are shipped in the repo, and the schema does not
depend on model quality:

.. code-block:: bash

   TASK=<your_task>          # set once, used by both commands

   lm_eval --model hf --model_args pretrained=facebook/opt-125m \
     --tasks "$TASK" --limit 3 --log_samples \
     --output_path "schema_check/$TASK"

   python -c "
   import json, glob, os, sys
   task = sys.argv[1]
   path = max(glob.glob(f'schema_check/{task}/**/samples_{task}_*.jsonl', recursive=True),
              key=os.path.getmtime)
   print('reading:', os.path.basename(path))
   row = json.loads(open(path).readline())
   print('top level:', sorted(row)); print('doc:', sorted(row['doc']))" "$TASK"

The task name belongs in both the output path and the glob, and the snippet
prints the file it read. A bare ``samples_*.jsonl`` glob over a directory shared
with earlier runs will hand you a different benchmark's rows, and nothing in the
output looks wrong — you just write your loader against the wrong schema.

Three things to take from the output:

- ``doc_id`` is the per-task item index and is the usual item ID.
- ``doc`` holds the raw dataset row. Your ``text=`` field comes from here.
- Each metric named in the YAML appears as a **top-level** key, not inside
  ``doc`` or ``metrics``.

``filter`` is always present, with the literal value ``"none"`` for tasks that
declare no ``filter_list``, so a missing ``filter_list`` does not mean a missing
``filter`` key.

Step 3 — Get the item count
---------------------------

``total_items`` is the one field neither the YAML nor a sample log gives you. It
needs a real count from lm-eval-harness, not a dataset card. Ask lm-eval:

.. code-block:: python

   from lm_eval.tasks import TaskManager

   td = TaskManager().load(["gsm8k"])
   print(sum(len(t.eval_docs) for t in td["tasks"].values()))
   # 1319

``td["tasks"]`` is keyed by task name and holds only leaf tasks, so the same sum
works for a task that expands into subtasks: ``load(["mmlu"])`` returns all 57
subject tasks and sums to 14042.

This materializes the dataset, so expect a minute or two and a download on first
call. It also resolves the split for you, which a dataset card will not: PIQA
ships 1838 validation and 3084 test rows, but its YAML sets ``test_split: null``,
so 1838 is the correct ``total_items``.


Step 4 — Register the benchmark
-------------------------------

Add an entry to ``BENCHMARKS`` in ``quark/contrib/mince/config.py``:

.. code-block:: python

   "gsm8k": BenchmarkConfig(
       name="gsm8k",
       total_items=1319,
       metric_names=["exact_match"],
       sample_glob="samples_gsm8k_*.jsonl",
       candidate_ns=[100, 200, 300, 400, 500, 600, 700, 800],
       sample_filter="flexible-extract",
   ),

The fields, and where each value comes from:

``name``
  Must equal the dict key. Used to look up the loader.

``total_items``
  The count from `Step 3 — Get the item count`_. Acts as a completeness guard:
  ``load_benchmark_items`` raises if the number of loaded items differs, which
  catches a truncated or partial log before it silently skews a sizing run.

``metric_names``
  Drawn from ``metric_list[].metric``, but these must match exactly the keys your
  loader writes into ``model_results`` — sizing and drift reporting iterate over
  this list and will ``KeyError`` on a name the loader never populates. Listing a
  subset of the YAML's metrics is fine: ARC declares ``acc`` and ``acc_norm``, and
  a config may size on ``acc`` alone provided the loader writes only ``acc``.

``sample_filter``
  The ``filter_list[].name`` value you picked from the task YAML. Omit it when the
  task declares no ``filter_list``; it defaults to ``""``, meaning every row is
  loaded. Your loader must honor it — see step 5.

``sample_glob``
  Glob matching lm-eval's output filenames, which are
  ``samples_<task>_<timestamp>.jsonl``.

``candidate_ns``
  The subset sizes swept during Monte-Carlo sizing, and the only values ``n*`` can
  be chosen from. Follow the shipped convention: 8 to 14 values spanning roughly
  10% to 60% of ``total_items``. Do not try to cover the whole dataset — drift at
  ``n == total_items`` is zero by definition and tells you nothing. ``run_sizing``
  raises outright if any candidate exceeds ``total_items``.

Step 5 — Write a loader
-----------------------

Add a loader to ``quark/contrib/mince/data_loader.py`` and register it in
``_LOADERS`` at the bottom of that file. Every loader takes the same two inputs
and returns the same type:

.. code-block:: python

   def _load_<name>(
       benchmark: BenchmarkConfig,        # the entry you just registered
       model_paths: dict[str, str],       # model_name -> dir holding sample JSONLs
   ) -> list[BenchmarkItem]: ...

Its job is to turn lm-eval's per-sample JSONL logs into one ``BenchmarkItem`` per
benchmark item, with every model's metric values attached to that item.

``model_paths`` is not configured anywhere — ``size.py`` builds it per run from
``--model-dirs``, mapping a label (the directory's basename) to a directory of
sample JSONLs. Those labels become the ``model_results`` keys. The CLI passes one
bf16 model today, but the signature is plural because sizing takes the worst P95
across every model it is given.

Four rules it must honor:

- Return the items in a fixed canonical order. The frozen subset is a list of
  *positions* into this list, so the order must be reproducible across runs — and
  across models, which is why the first model fixes the order below and the rest
  attach by ID lookup rather than by file order.
- Populate ``model_results[model_name]`` for every model in ``model_paths``,
  keyed by the names in ``metric_names``.
- Skip rows whose ``filter`` does not match ``benchmark.sample_filter``. Forget
  this on a filtered task and you load one item per filter variant, which trips
  the ``total_items`` guard with an error that does not name the cause.
- Raise on a duplicate item ID rather than overwriting.

This is the real ``_load_gsm8k``, abridged. It needs ``json``, ``BenchmarkItem``
and ``_find_sample_file`` — all already imported or defined in
``data_loader.py``. ``_find_sample_file`` resolves a directory plus a glob to one
path, preferring the newest file when several match:

.. code-block:: python

   def _load_gsm8k(
       benchmark: BenchmarkConfig,
       model_paths: dict[str, str],
   ) -> list[BenchmarkItem]:
       """Load GSM8K items from one sample JSONL per model.

       lm-eval logs one row per filter, so GSM8K writes two rows per item; rows are
       kept only when they match ``benchmark.sample_filter``. Items are keyed by
       ``doc_id`` with a single ``exact_match`` metric.

       Args:
           benchmark: The GSM8K benchmark configuration, including ``sample_filter``.
           model_paths: Maps model_name -> directory containing the sample JSONL.

       Returns:
           List of BenchmarkItem in the first model's file order, each carrying
           per-model metrics in ``item.model_results[model_name]``.
       """
       first_model = next(iter(model_paths))
       first_path = _find_sample_file(model_paths[first_model], benchmark.sample_glob)

       sample_filter = benchmark.sample_filter
       canonical_ids: list[int] = []
       items_by_id: dict[int, BenchmarkItem] = {}

       # First model fixes the canonical order.
       with open(first_path) as f:
           for line in f:
               doc = json.loads(line)
               if sample_filter and doc.get("filter") != sample_filter:
                   continue
               doc_id = doc["doc_id"]
               if doc_id in items_by_id:
                   raise ValueError(f"{benchmark.name}: duplicate canonical id {doc_id!r} in {first_path}")
               canonical_ids.append(doc_id)
               items_by_id[doc_id] = BenchmarkItem(
                   item_id=doc_id,
                   text=doc["doc"]["question"],
               )

       for model_name, model_dir in model_paths.items():
           path = _find_sample_file(model_dir, benchmark.sample_glob)
           with open(path) as f:
               for line in f:
                   doc = json.loads(line)
                   if sample_filter and doc.get("filter") != sample_filter:
                       continue
                   items_by_id[doc["doc_id"]].model_results[model_name] = {
                       "exact_match": doc.get("exact_match", doc.get("acc", 0.0)),
                   }

       return [items_by_id[did] for did in canonical_ids]

Then add one entry to the existing ``_LOADERS`` dict — leave the other entries
alone:

.. code-block:: python

   _LOADERS = {
       # ... existing entries ...
       "gsm8k": _load_gsm8k,
   }

If the task writes one file per subject instead, model the loader on
``_load_mmlu``: glob all files, derive the subject from each filename with
``_extract_subject(basename, "samples_mmlu_")``, and use a ``(subject, doc_id)``
tuple as the item ID, since ``doc_id`` restarts at zero in each subtask's file.

.. note::

   ``BenchmarkItem.stratum`` is an optional label logged in the frozen artifact
   for benchmarks with natural categories. It is never used for sampling, and
   leaving it unset is fine.

Step 6 — Add the ``--samples`` mapping
--------------------------------------

This step is required and is easy to miss, because nothing in steps 1-5 fails
without it. ``freeze.py`` translates a frozen artifact into lm-eval's
``--samples`` map via ``build_samples`` in ``quark/contrib/mince/subset.py``,
which dispatches on a hardcoded set. Add your benchmark to whichever of the two
fits — these grow with every benchmark, so expect more entries than shown:

.. code-block:: python

   # Frozen indices map 1:1 onto a single lm-eval task's doc order.
   _SINGLE_FILE = {"gsm8k", "ifeval", ...}

   # Group tasks: (subtask name prefix, artifact key holding the grouping field).
   _GROUP_SUBTASK = {
       "mmlu": ("mmlu_", "subject"),
       "mmlu_pro": ("mmlu_pro_", "category"),
   }

Both ``build_samples`` and the module docstring at the top of ``data_loader.py``
carry a per-benchmark inventory in prose. Update those too, or you will leave
them stale.

Skip this and freezing appears to succeed, then fails when the subset is written:

.. code-block:: text

   ValueError: no --samples mapping for benchmark 'my_benchmark': add it to
   _SINGLE_FILE or _GROUP_SUBTASK in quark/contrib/mince/subset.py

Then add a branch to ``build_item_id`` in the same file, so the artifact records
readable identifiers:

.. code-block:: python

   if benchmark_name in ("gsm8k", "my_benchmark"):
       return {"doc_id": item.item_id, "question": item.text[:80]}

These keys are only labels in the frozen artifact, so join the branch above only
if ``question`` actually describes your item text. A task whose text is not a
question — PIQA's is a ``goal`` — is better off with its own branch and its own
key, or the artifact reads as if it holds something it does not.

.. warning::

   ``build_item_id`` has a catch-all fallback that returns
   ``{"item_id": str(item.item_id)}``, so an unregistered benchmark does **not**
   raise here. It silently writes a frozen artifact with the ID stringified and
   the item text dropped. Unlike step 6's first half, nothing will tell you.

To check your work after a real freeze, ``freeze.py`` writes
``frozen_subset_seed<seed>.json`` and ``subset_samples.json`` into
``mince_frozen/<benchmark>/<model-label>/`` unless ``--out-dir`` says otherwise.
Every entry under ``items`` should carry your keys, not ``item_id``.

Step 7 — Add a fixture and tests
--------------------------------

Fixtures build tiny synthetic logs, so the tests need no model and no network.
``Fixture`` (an alias for ``tuple[BenchmarkConfig, dict[str, str]]``) and
``write_jsonl`` both come from ``quark/contrib/mince/test/utils.py``. Give the
fixture and both tests docstrings, as the rest of the suite does.

Add a fixture to ``quark/contrib/mince/test/conftest.py``. Note that it builds
its own small ``BenchmarkConfig`` rather than importing the real one from
``BENCHMARKS`` — the real ``total_items`` would fail the completeness guard
against six synthetic rows:

.. code-block:: python

   @pytest.fixture
   def gsm8k_data(tmp_path: Path) -> Fixture:
       """Provide a 6-item synthetic GSM8K benchmark with two filter rows per item.

       Args:
           tmp_path: pytest's per-test temporary directory.

       Returns:
           A ``(config, model_paths)`` pair ready for ``load_benchmark_items``.
       """
       config = BenchmarkConfig(
           name="gsm8k",
           total_items=6,
           metric_names=["exact_match"],
           sample_glob="samples_gsm8k_*.jsonl",
           candidate_ns=[2, 3, 4],
           sample_filter="flexible-extract",
       )
       model_paths = {}
       for mi, model in enumerate(["modelA"]):
           d = os.path.join(tmp_path, "gsm8k", model)
           rows = []
           for i in range(6):
               base = {"doc_id": i, "doc": {"question": f"gsm q{i}"}}
               # A second filter row that the flexible-extract filter must drop.
               rows.append({**base, "filter": "strict-match", "exact_match": 0.0})
               rows.append({**base, "filter": "flexible-extract", "exact_match": float((i + mi) % 2)})
           write_jsonl(os.path.join(d, "samples_gsm8k_2026-01-01T00-00-00.jsonl"), rows)
           model_paths[model] = d
       return config, model_paths

Then add two tests. The first, in ``test_data_loader.py``, asserts the three
things you are responsible for — item count, canonical order, and metric keys:

.. code-block:: python

   def test_gsm8k_applies_flexible_extract_filter(gsm8k_data: Fixture) -> None:
       """GSM8K yields one item per doc, dropping rows from the non-matching filter."""
       config, model_paths = gsm8k_data
       items = load_benchmark_items(config, model_paths)

       # 6 docs, two filter rows each -> only flexible-extract kept.
       assert len(items) == 6
       assert [it.item_id for it in items] == list(range(6))
       assert all("exact_match" in it.model_results["modelA"] for it in items)

The second, in ``test_build_samples.py``, covers step 6, which no other test
reaches. That file imports ``build_samples as _build_samples``:

.. code-block:: python

   def test_my_benchmark_single_file(my_benchmark_data: Fixture) -> None:
       """My benchmark maps to flat indices with ``doc_id``-enriched frozen items."""
       config, model_paths = my_benchmark_data
       artifact, _ = build_frozen_subset(config, model_paths, n=3, seed=42)

       samples = _build_samples(artifact)
       assert set(samples) == {config.name}
       assert samples[config.name] == artifact["indices"]
       # Not the stringified fallback from build_item_id.
       assert all("doc_id" in it for it in artifact["items"])

Finally, add your fixture to ``test_sizing_runs_for_all_simple_benchmarks`` in
``test_smoke_core.py``. The test takes its fixtures explicitly
rather than enumerating the registry, so it does not cover a new benchmark until
you add one. It exercises the sizing sweep end to end: that every value in
``candidate_ns`` is in range, and that ``metric_names`` survives the vectorized
aggregation in ``run_model``.


Metrics that are not a per-item mean
------------------------------------

A metric whose YAML declares an ``!function`` aggregation instead of ``mean`` is
not a per-item average, so ``_compute_simple_metrics`` does not apply.
**IFEval is the shipped example. Please follow how it is wired.**

.. warning::

   It has to be wired in *two* places: ``compute_accuracy`` in ``mince_metrics.py``
   dispatches the exact scorer used at freeze time, and ``run_model`` in
   ``montecarlo.py`` dispatches a vectorized twin used by the sizing sweep, which
   runs thousands of iterations. They are separate implementations of the same
   aggregation. Add one without the other and sizing and freezing will silently
   report different accuracies for the same subset.

Workflow summary
----------------

.. code-block:: text

   1. read the task YAML          -> metric_names, sample_filter, aggregation
   2. log 3 samples w/ opt-125m   -> JSONL keys for item_id, text, metrics
   3. count eval_docs             -> total_items
   4. config.py                   -> BenchmarkConfig entry in BENCHMARKS
   5. data_loader.py              -> loader + _LOADERS entry
   6. subset.py                   -> _SINGLE_FILE/_GROUP_SUBTASK + build_item_id
   7. test/                       -> fixture + loader test + build_samples test

Steps 4 through 6 are all required for a working benchmark. With those done, the
benchmark is selectable everywhere: ``size.py`` and ``freeze.py`` derive their
``--benchmark`` choices from ``BENCHMARKS.keys()``, and ``validate.py`` takes the
task name as a free-form string. Sizing, freezing, and validating then follow the
standard flow in :doc:`example_quark_torch_mince`.

If sizing reports drift that never settles below your budget across the whole
``candidate_ns`` range, widen the range upward before concluding the benchmark is
unsuitable — ``n*`` can only be chosen from the values you swept.
