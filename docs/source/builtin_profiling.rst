Built-in Profiling
==================

Overview
--------

Quark includes a built-in profiler that allows users to track, among other things, CPU memory usage (RSS) during the quantization process. This feature helps identify memory bottlenecks and understand the resource consumption of different quantization steps.

Unlike external tools that sample memory periodically, this built-in profiler is integrated into the Quark codebase and records memory usage at specific execution steps (e.g., "Start", "Model Loaded", "Quantization Finished"), providing a more context-aware view of memory consumption.

**Two-Level Profiling Architecture:**

1. **Internal Collection Points**: Quark automatically profiles key quantization stages (Pre-process, Calibration, Quantization, Post-process, etc.) without any user code changes. These measurements happen inside Quark's quantization functions.

2. **User-Defined Checkpoints**: Users can add application-level profiling in their scripts (e.g., data loading, model preparation, export) using the same profiler API.

This design makes profiling both easy to use (automatic for core operations) and flexible (extensible for custom workflows).

The profiler uses an extensible metric system. Default metrics are registered automatically, but users can add custom metrics by extending the ``SummaryMetric`` or ``CheckpointMetric`` base classes.

Quick Start
-----------

**Automatic Profiling (Most Common):**

For ONNX or PyTorch quantization workflows, simply enable profiling - Quark automatically profiles all quantization stages:

.. code-block:: python

    import os
    os.environ["QUARK_PROFILING"] = "1"

    from quark.onnx import quantize_static
    # or: from quark.torch import ModelQuantizer

    # Run your quantization - profiling happens automatically!
    quantize_static(...)  # Results written to quark_profile.yaml

**User-Defined Checkpoints (Optional):**

Add application-level profiling for operations outside Quark's internal quantization:

.. code-block:: python

    import os
    os.environ["QUARK_PROFILING"] = "1"

    from quark.common.profiler import GlobalProfiler, ProfileStep

    profiler = GlobalProfiler(output_path="quark_profile.yaml")

    # Use predefined constants for common operations
    with profiler.scope(ProfileStep.MODEL_LOADING):
        model = load_model()  # Your custom model loading

    # Use USER_DEFINED with user_msg for custom operations
    with profiler.scope(ProfileStep.USER_DEFINED, user_msg="Custom Data Preprocessing"):
        preprocess_data()  # Your custom preprocessing

    # Results written to quark_profile.yaml on exit

Enabling the Profiler
---------------------

To enable the profiler, simply set the environment variable ``QUARK_PROFILING`` to ``1`` before running your quantization script.

.. code-block:: bash

    export QUARK_PROFILING=1

or in Python:

.. code-block:: python

    import os
    os.environ["QUARK_PROFILING"] = "1"

Usage in Code
-------------

The profiler provides three usage patterns:

**Pattern 1: Context Manager (Recommended)**

.. code-block:: python

    from quark.common.profiler import GlobalProfiler, ProfileStep

    profiler = GlobalProfiler(output_path="my_profile.yaml")

    with profiler.scope(ProfileStep.MODEL_QUANTIZATION):
        quantized_model = quantize(model)

**Pattern 2: Decorator**

.. code-block:: python

    from quark.common.profiler import profile_scope, ProfileStep

    @profile_scope(ProfileStep.CALIBRATION)
    def calibrate_model(model, data):
        # calibration code
        return calibrated_model

    # For custom operations, use USER_DEFINED with user_msg
    @profile_scope(ProfileStep.USER_DEFINED, user_msg="Custom Validation")
    def validate_model(model):
        # custom validation logic
        return validation_results

**Pattern 3: Multiple Custom Steps**

You can use ``ProfileStep.USER_DEFINED`` with different messages for various custom operations:

.. code-block:: python

    from quark.common.profiler import GlobalProfiler, ProfileStep

    profiler = GlobalProfiler(output_path="my_profile.yaml")

    # Profile different custom operations
    with profiler.scope(ProfileStep.USER_DEFINED, user_msg="Custom Data Validation"):
        validate_data()

    with profiler.scope(ProfileStep.USER_DEFINED, user_msg="Custom Feature Engineering"):
        engineer_features()

    with profiler.scope(ProfileStep.USER_DEFINED, user_msg="Custom Post-processing"):
        postprocess_results()

ProfileStep Constants
---------------------

The ``ProfileStep`` class defines standard constants for profiling steps across both PyTorch and ONNX quantization workflows. Use these constants with ``profiler.scope()`` to ensure consistent naming and enable IDE auto-complete.

**Note on Internal vs. User-Defined Collection:**

* **Automatically collected by Quark** (marked with ✓): These steps are profiled internally by Quark's quantization functions. Users do not need to add these checkpoints manually.
* **User-defined in scripts**: Constants provided for user code to profile application-level operations (data loading, model export, etc.).

**General Constants:**

* ``ProfileStep.START``: "Start" - ✓ Automatically created at profiler initialization
* ``ProfileStep.END``: "End" - ✓ Automatically created when profiler stops
* ``ProfileStep.USER_DEFINED``: "User-Defined" - For custom profiling with ``user_msg`` parameter
* ``ProfileStep.MODEL_LOADED``: "Model Loaded" - User-defined

**PyTorch Quantization Steps:**

* ``ProfileStep.FILE_TO_FILE_QUANTIZATION``: "File-to-File Quantization" - ✓ Auto-collected (when using file-to-file API)
* ``ProfileStep.MODEL_LOADING``: "Model Loading" - User-defined
* ``ProfileStep.DATASET_LOADING``: "Dataset Loading" - User-defined
* ``ProfileStep.MODEL_QUANTIZATION``: "Model Quantization" - ✓ Auto-collected (by ``ModelQuantizer.quantize_model()``)
* ``ProfileStep.MODEL_EVALUATION``: "Model Evaluation" - User-defined
* ``ProfileStep.CALIBRATION``: "Calibration" - ✓ Auto-collected (during calibration phase)
* ``ProfileStep.CALIBRATION_WEIGHTS``: "Calibration (Weights)" - Internal sub-step
* ``ProfileStep.CALIBRATION_FORWARD``: "Calibration (Forward)" - Internal sub-step
* ``ProfileStep.FREEZE_MODEL``: "Freeze Model" - ✓ Auto-collected (by ``ModelQuantizer.freeze()``)
* ``ProfileStep.MODEL_PREPARATION``: "Model Preparation" - ✓ Auto-collected (during model prep)
* ``ProfileStep.ADVANCED_ALGORITHMS``: "Advanced Algorithms" - ✓ Auto-collected (when using advanced algorithms)

**Export Steps (PyTorch):**

* ``ProfileStep.EXPORT_HF_SAFETENSORS``: "Export HF Safetensors" - User-defined
* ``ProfileStep.EXPORT_ONNX``: "Export ONNX" - User-defined
* ``ProfileStep.EXPORT_GGUF``: "Export GGUF" - User-defined

**ONNX Quantization Steps:**

* ``ProfileStep.PRE_PROCESS``: "Pre-process" - ✓ Auto-collected (by ONNX preprocessing)
* ``ProfileStep.CALIBRATION``: "Calibration" - ✓ Auto-collected (by ONNX calibration)
* ``ProfileStep.QUANTIZATION_MATMUL_NBITS``: "Quantization (MatMulNBits)" - ✓ Auto-collected (by MatMulNBits quantizer)
* ``ProfileStep.QUANTIZATION_STATIC``: "Quantization (Static)" - ✓ Auto-collected (by static quantizer)
* ``ProfileStep.QUANTIZATION_DYNAMIC``: "Quantization (Dynamic)" - ✓ Auto-collected (by dynamic quantizer)
* ``ProfileStep.POST_PROCESS``: "Post-process" - ✓ Auto-collected (by ONNX postprocessing)
* ``ProfileStep.FAST_FINETUNE``: "Fast Finetune" - Internal (when finetuning enabled)
* ``ProfileStep.MODEL_CACHING``: "Model Caching" - Internal (when caching enabled)
* ``ProfileStep.FLOAT_MODEL_VALIDATION``: "Float Model Validation" - Internal (when validation enabled)

**Automatic Start/End Checkpoints:**

When used with ``profiler.scope()``, the profiler automatically creates "Start" and "End" checkpoints:

.. code-block:: python

    with profiler.scope(ProfileStep.MODEL_LOADING):
        # code here
    # Creates: "Model Loading Start" and "Model Loading End" checkpoints

Results
-------

The profiling results are saved to a YAML file (default: ``quark_profile.yaml``). The file contains the total time and a list of records for each checkpoint.

Example Output:

.. code-block:: yaml

    # Quark Profiling Results
    # Checkpoints are written in real-time as they occur

    memory_usage:
    - step: "Start"
      timestamp: 1709395200.123
      relative_time_secs: 0.0
      cpu_memory_mb: 150.5
      gpu_memory_mb: 0.0
      disk_read_mb: 0.0
      disk_write_mb: 0.0
    - step: "Model Loading Start"
      timestamp: 1709395200.456
      relative_time_secs: 0.333
      cpu_memory_mb: 155.2
      gpu_memory_mb: 0.0
      disk_read_mb: 12.5
      disk_write_mb: 1.2
    - step: "Model Loading End"
      timestamp: 1709395205.789
      relative_time_secs: 5.666
      cpu_memory_mb: 2048.0
      gpu_memory_mb: 1024.5
      disk_read_mb: 1024.0
      disk_write_mb: 3.4
    - step: "Model Quantization Start"
      timestamp: 1709395206.012
      relative_time_secs: 5.889
      cpu_memory_mb: 2048.5
      gpu_memory_mb: 1025.0
      disk_read_mb: 1030.1
      disk_write_mb: 5.0
    - step: "Model Quantization End"
      timestamp: 1709395245.678
      relative_time_secs: 45.555
      cpu_memory_mb: 1536.0
      gpu_memory_mb: 768.0
      disk_read_mb: 1245.7
      disk_write_mb: 512.3
    - step: "End"
      timestamp: 1709395245.901
      relative_time_secs: 45.778
      cpu_memory_mb: 1530.0
      gpu_memory_mb: 765.0
      disk_read_mb: 1246.0
      disk_write_mb: 513.1

    # Summary Metrics
    total_quantization_time_seconds: 45.778
    peak_memory_mb: 2048.5
    gpu_peak_memory_mb: 1025.0
    total_disk_read_mb: 1246.0
    total_disk_write_mb: 513.1

    # Metric Definitions:
    #
    # Checkpoint Metrics (per record):
    # - step: Name of the profiling checkpoint
    # - timestamp: Unix timestamp (seconds since epoch)
    # - relative_time_secs: Time elapsed since profiling started
    # - cpu_memory_mb: Current RSS memory in megabytes
    # - gpu_memory_mb: Current GPU memory usage in megabytes
    # - disk_read_mb: Cumulative disk bytes read (MB) since the start of profiling
    # - disk_write_mb: Cumulative disk bytes written (MB) since the start of profiling
    #
    # Summary Metrics (overall):
    # - total_quantization_time_seconds: Total elapsed time from start to end
    # - peak_memory_mb: Peak CPU memory (RSS) during profiling session
    # - peak_gpu_memory_mb: Peak GPU memory during profiling session
    # - total_disk_read_mb: Total disk bytes read (MB) during the entire profiling session
    # - total_disk_write_mb: Total disk bytes written (MB) during the entire profiling session

Output Format Details
~~~~~~~~~~~~~~~~~~~~~

The YAML output file consists of global statistics and a detailed log of memory events.

**Summary Metrics (top-level):**

* ``total_quantization_time_seconds``: The total duration (in seconds) from the start to the stop of the profiler.
* ``peak_memory_mb``: The maximum Resident Set Size (RSS) memory reached by the process during the profiled session. This provides an upper bound on memory consumption.
* ``gpu_peak_memory_mb``: The maximum GPU memory usage in MB during the profiled session (when GPU profiling is enabled).
* ``total_disk_read_mb``: Total disk bytes read (in MB) by the process and its children during the entire profiling session, measured relative to the baseline captured at start (when disk I/O profiling is enabled).
* ``total_disk_write_mb``: Total disk bytes written (in MB) by the process and its children during the entire profiling session, measured relative to the baseline captured at start (when disk I/O profiling is enabled).

**Checkpoint Metrics (within ``memory_usage``):**

This list contains snapshots recorded at each ``checkpoint()`` call.

* ``step``: A descriptive string label for the quantization step or event (e.g., "Start", "Calibrate Method").
* ``timestamp``: The absolute system timestamp (Unix epoch) of the event.
* ``relative_time_secs``: Time in seconds elapsed since the profiler started. Use this to analyze the timing of sequential steps.
* ``cpu_memory_mb``: Current CPU memory usage (RSS) in Megabytes (MB).
* ``gpu_memory_mb``: Current GPU memory usage in Megabytes (MB) (when GPU profiling is enabled).
* ``disk_read_mb``: Cumulative disk bytes read (in MB) since the start of profiling, measured relative to the baseline captured at the ``Start`` checkpoint. Covers the main process and all child processes (when disk I/O profiling is enabled).
* ``disk_write_mb``: Cumulative disk bytes written (in MB) since the start of profiling, measured relative to the baseline captured at the ``Start`` checkpoint. Covers the main process and all child processes (when disk I/O profiling is enabled).

**Metric Definitions in Output:**

The YAML output automatically includes detailed metric definitions at the bottom as comments. These definitions are dynamically generated from each metric's ``get_definition()`` method, so custom metrics will automatically have their definitions included in the output file.

GPU Memory Profiling
--------------------

The GlobalProfiler automatically tracks GPU memory usage when GPUs are available. Both NVIDIA CUDA and AMD ROCm platforms are supported.

**Platform Support:**

* **CUDA (NVIDIA GPUs)**: Uses ``nvidia-smi`` to query GPU memory for devices in ``CUDA_VISIBLE_DEVICES``
* **ROCm (AMD GPUs)**: Uses ``rocm-smi`` to query GPU memory for devices in ``CUDA_VISIBLE_DEVICES`` or ``HIP_VISIBLE_DEVICES``

**Automatic GPU Detection:**

GPU profiling is enabled automatically when GPUs are detected. The profiler will:

* Profile all available GPUs if no environment variables are set
* Profile specific GPUs when ``CUDA_VISIBLE_DEVICES`` or ``HIP_VISIBLE_DEVICES`` are set
* Skip GPU profiling if no GPUs are available (CPU-only mode)

**Best Practices for Accurate GPU Profiling:**

For the most accurate GPU memory measurements, it is recommended to set ``CUDA_VISIBLE_DEVICES`` (or ``HIP_VISIBLE_DEVICES`` for ROCm) to isolate specific GPU(s). This prevents the profiler from including memory usage from other processes on shared GPUs:

.. code-block:: bash

    # For NVIDIA GPUs (recommended)
    export CUDA_VISIBLE_DEVICES=0
    export QUARK_PROFILING=1
    python your_script.py

    # For AMD GPUs (recommended)
    export CUDA_VISIBLE_DEVICES=0  # or HIP_VISIBLE_DEVICES=0
    export QUARK_PROFILING=1
    python your_script.py

If these environment variables are not set, the profiler will issue a warning and track all available GPUs. This may include memory from other processes, potentially leading to less accurate measurements.

**GPU Metrics:**

When GPU profiling is enabled, the following metrics are tracked:

* ``gpu_memory_mb``: GPU memory usage at each checkpoint (in checkpoint records)
* ``gpu_peak_memory_mb``: Peak GPU memory usage across all checkpoints (in summary metrics)

If no GPUs are available, these metrics will not appear in the output YAML file.

Disk I/O Profiling
-------------------

The GlobalProfiler automatically tracks disk read and write activity for the current process and all its children. This is useful for understanding the I/O cost of model loading, checkpoint saving, calibration data loading, and other file-intensive operations.

**Platform Support:**

Disk I/O profiling relies on ``psutil``'s ``Process.io_counters()`` API, which in turn reads per-process I/O statistics from the operating system:

* **Linux**: Available via ``/proc/<pid>/io``. No special privileges required.
* **Windows**: Available natively.
* **macOS**: Requires root privileges. Without root, disk I/O metrics will be omitted from the output YAML.

If disk I/O counters are not available on the current platform, ``disk_read_mb``, ``disk_write_mb``, ``total_disk_read_mb``, and ``total_disk_write_mb`` will all be absent from the output file.

**How the Baseline Works:**

When profiling starts, the profiler records the cumulative I/O counters already accumulated by the process as a *baseline*. All subsequent measurements subtract this baseline, so the metrics reflect only the I/O activity that occurred *during* the profiling session — not any I/O from before ``GlobalProfiler`` was initialized.

**Disk Metrics:**

* ``disk_read_mb``: Cumulative disk bytes read (in MB) since the baseline, reported at each checkpoint.
* ``disk_write_mb``: Cumulative disk bytes written (in MB) since the baseline, reported at each checkpoint.
* ``total_disk_read_mb``: Total disk bytes read (in MB) across the entire profiling session (summary metric).
* ``total_disk_write_mb``: Total disk bytes written (in MB) across the entire profiling session (summary metric).

.. note::

   On Linux, ``/proc/<pid>/io`` counters include all bytes passed through the kernel's read/write syscalls, even if served from the page cache. This means ``disk_read_mb`` may be higher than the actual physical bytes transferred from storage.

Framework-Specific Usage
------------------------

PyTorch Example
~~~~~~~~~~~~~~~

Complete example for PyTorch model quantization showing user-defined checkpoints:

.. code-block:: python

    import os
    os.environ["QUARK_PROFILING"] = "1"

    from quark.common.profiler import GlobalProfiler, ProfileStep
    from quark.torch import ModelQuantizer

    # Initialize profiler
    profiler = GlobalProfiler(output_path="torch_profile.yaml")

    # User-defined: Profile model loading
    with profiler.scope(ProfileStep.MODEL_LOADING):
        model = load_pretrained_model()

    # User-defined: Profile dataset loading
    with profiler.scope(ProfileStep.DATASET_LOADING):
        calib_dataloader = get_calibration_data()

    # Quark auto-profiles: MODEL_QUANTIZATION, CALIBRATION, FREEZE_MODEL, etc.
    quantizer = ModelQuantizer(quant_config)
    quantized_model = quantizer.quantize_model(model, calib_dataloader)
    quantized_model = quantizer.freeze(quantized_model)

    # User-defined: Profile export step
    with profiler.scope(ProfileStep.EXPORT_ONNX):
        export_onnx(quantized_model, "model.onnx")

    # Results automatically saved to torch_profile.yaml
    # Output includes both user-defined and Quark's internal checkpoints

ONNX Example
~~~~~~~~~~~~

Complete example for ONNX model quantization:

.. code-block:: python

    import os
    os.environ["QUARK_PROFILING"] = "1"

    from quark.common.profiler import GlobalProfiler, ProfileStep
    from quark.onnx import quantize_static

    # Initialize profiler
    profiler = GlobalProfiler(output_path="onnx_profile.yaml")

    # User could add custom checkpoints for data preparation if needed
    # with profiler.scope("Data Preprocessing"):
    #     calibration_data_reader = prepare_calibration_data()

    # Quark auto-profiles: PRE_PROCESS, CALIBRATION, QUANTIZATION_STATIC, POST_PROCESS
    quantize_static(
        model_input="float_model.onnx",
        model_output="quantized_model.onnx",
        calibration_data_reader=calibration_data_reader,
    )

    # Results automatically saved to onnx_profile.yaml
    # Output includes Quark's internal checkpoints (no user intervention needed)

**Automatic Internal Profiling:**

When you call ONNX quantization functions, Quark automatically profiles these steps:

* ``PRE_PROCESS`` - Model preprocessing
* ``CALIBRATION`` - Calibration data collection
* ``QUANTIZATION_STATIC`` / ``QUANTIZATION_DYNAMIC`` / ``QUANTIZATION_MATMUL_NBITS`` - Quantization phase
* ``POST_PROCESS`` - Model postprocessing

You only need to add user-defined checkpoints for your application-level operations.

For more detailed ONNX profiling examples and workflow-specific guidance, see :doc:`onnx/tutorial_profiling`.

Creating Custom Metrics
-----------------------

The profiler supports custom metrics through an extensible class hierarchy. You can create metrics that are collected either at the end of profiling (summary metrics) or at each checkpoint (checkpoint metrics).

Metric Types
~~~~~~~~~~~~

There are two types of metrics:

* **SummaryMetric**: Collected once when ``stop()`` is called. Use for aggregate or final values like total time or peak memory.
* **CheckpointMetric**: Collected at each ``checkpoint()`` call. Use for point-in-time measurements like current memory usage.

Creating a Custom Checkpoint Metric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create a custom checkpoint metric, extend the ``CheckpointMetric`` class and implement the required methods:

.. code-block:: python

    from quark.common.profiler import (
        GlobalProfiler,
        CheckpointMetric,
        MetricContext,
    )

    class GpuMemoryMetric(CheckpointMetric):
        """Custom metric to track GPU memory usage."""

        @property
        def name(self) -> str:
            """The YAML key name for this metric."""
            return "gpu_memory_mb"

        def get_definition(self) -> str:
            """Human-readable description for the YAML comments."""
            return "GPU memory usage in MB (requires pynvml)."

        def collect(self, context: MetricContext) -> float | None:
            """Collect the metric value. Return None to skip."""
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                return info.used / (1024 * 1024)  # Convert to MB
            except Exception:
                return None  # Skip this metric if unavailable

    # Register and use the custom metric
    profiler = GlobalProfiler()
    profiler.checkpoint_metrics.append(GpuMemoryMetric())
    profiler._checkpoint("GPU Operation")

Creating a Custom Summary Metric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create a custom summary metric, extend the ``SummaryMetric`` class:

.. code-block:: python

    from quark.common.profiler import (
        GlobalProfiler,
        SummaryMetric,
        MetricContext,
    )

    class ModelSizeMetric(SummaryMetric):
        """Custom metric to report the model size."""

        def __init__(self, model_path: str):
            self.model_path = model_path

        @property
        def name(self) -> str:
            return "model_size_mb"

        def get_definition(self) -> str:
            return "Size of the quantized model file in MB."

        def collect(self, context: MetricContext) -> float | None:
            import os
            if os.path.exists(self.model_path):
                size_bytes = os.path.getsize(self.model_path)
                return size_bytes / (1024 * 1024)
            return None

    # Register and use the custom metric
    profiler = GlobalProfiler()
    profiler.summary_metrics.append(ModelSizeMetric("output_model.onnx"))
    # ... quantization ...
    # Results written on exit

MetricContext
~~~~~~~~~~~~~

The ``MetricContext`` dataclass provides all the information a metric might need:

* ``start_time``: Unix timestamp when profiling started.
* ``end_time``: Unix timestamp when profiling stopped (only for summary metrics).
* ``base_memory``: Baseline CPU memory in bytes recorded at start.
* ``process``: The ``psutil.Process`` object (may be None).
* ``step_name``: Name of the current checkpoint (only for checkpoint metrics).
* ``current_time``: Unix timestamp of the current checkpoint.
* ``current_memory``: Current CPU memory usage in bytes.
* ``gpu_available``: Whether GPU profiling is enabled.
* ``gpu_baseline_memory``: Baseline GPU memory in bytes.
* ``gpu_current_memory``: Current GPU memory in bytes (only for checkpoint metrics).
* ``gpu_peak_memory``: Peak GPU memory in bytes (only for summary metrics).
* ``disk_io_available``: Whether disk I/O counters are available via ``psutil`` on this platform (Linux and Windows; not available on macOS without root).
* ``baseline_disk_read_bytes``: Cumulative bytes read by the process at the start of profiling.
* ``baseline_disk_write_bytes``: Cumulative bytes written by the process at the start of profiling.
* ``current_disk_read_bytes``: Cumulative bytes read at this checkpoint (only for checkpoint metrics).
* ``current_disk_write_bytes``: Cumulative bytes written at this checkpoint (only for checkpoint metrics).

Requirements
------------

The profiler requires additional dependencies.

* **psutil**: Required for fetching memory statistics.
* **PyYAML**: Recommended for writing the YAML output.

You can install these dependencies using the ``profiling`` extra:

.. code-block:: bash

    pip install .[profiling]

Or install them individually:

.. code-block:: bash

    pip install psutil PyYAML
