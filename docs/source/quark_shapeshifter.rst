.. Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

Quark Shapeshifter
==================

**AMD Quark** includes a powerful graph transformation tool called `Shapeshifter`. It provides a rich set of transformation passes for both **ONNX models** and **PyTorch models**:

* **For ONNX models:** Preprocessing passes (constant folding, operator fusion, BatchNorm folding, redundant-node removal, input/output streamlining) to prepare models before quantization, and postprocessing passes (Q/DQ cleanup, scale alignment, bfloat16 adaptation, XINT8/NPU simulation) to tailor quantized models for specific hardware deployment targets.

* **For PyTorch models:** Transformation passes (layer removal, model tracing, pre-export preparation) to optimize PyTorch models directly without converting to ONNX, enabling use cases like model structure transformations, removing inference-unnecessary layers, and preparing models for export.

The adapter supports both **file-based workflows** (loading and saving models from/to disk) and **in-memory workflows** (operating on Python objects directly) via its programmatic API.


Architecture Overview
---------------------

Shapeshifter is a **pass-based transformation framework** that enables modular, composable model transformations through a plugin-style architecture.

How It Works
~~~~~~~~~~~~

**1. Pass-Based Design**

Each transformation is encapsulated in a **pass** - a self-contained unit that:

* Takes a model as input
* Applies one specific transformation (e.g., fold BatchNorm, remove dropout, convert opset)
* Returns the modified model

Passes are **composable**: chain multiple passes together to build complex transformation pipelines. Each pass operates on the output of the previous one.

**2. Automatic Pass Discovery and Registration**

The system uses a plugin-style architecture for zero-configuration pass loading:

* **Core Passes**: Automatically discovered from ``quark/shapeshifter/passes/``
* **Community Passes**: Automatically discovered from ``quark/contrib/shapeshifter_community_passes/``
* **Zero Configuration**: Just add a ``.py`` file with a ``@register_pass`` decorated class - no manual registration needed
* **Unified Registry**: All passes register to a single global ``REGISTRY`` using the filename as the pass name (e.g., ``pytorch_remove_dropout.py`` → ``pytorch_remove_dropout``)

**3. Core vs Community Passes**

* **Core Passes**: Officially maintained, stable, production-ready transformations
* **Community Passes**: Contributed passes that extend Shapeshifter's capabilities
* **Seamless Integration**: Both types work together - users can freely mix core and community passes in their workflows
* **Optional Loading**: Community passes are optional

**4. Type Safety and Validation**

* All passes are either ONNX passes (``ONNXPass``) or PyTorch passes (``PytorchPass``)
* A workflow can only contain one type - mixing ONNX and PyTorch passes raises ``ValueError``

**5. Flexible Workflows**

Shapeshifter supports multiple execution modes:

* **CLI (File-based)**: ``quark-cli shapeshifter config.yaml`` - Load model from disk, apply passes, save result
* **API (File-based)**: Same as CLI but called from Python code
* **API (In-memory)**: Pass model objects directly - no I/O overhead, ideal for programmatic use

For detailed technical information about design patterns, data flow, and implementation details, see the "Architecture and Design (For Developers)" section below.

Available Passes
----------------

Shapeshifter supports two categories of passes: **ONNX passes** for transforming ONNX graph models, and **PyTorch passes** for transforming PyTorch models. All passes in a single workflow must be of the same type (all ONNX or all PyTorch).

.. toctree::
   :maxdepth: 2
   :caption: Pass Documentation

   quark_shapeshifter_onnx_passes
   quark_shapeshifter_torch_passes

Using Shapeshifter
------------------

The Shapeshifter supports both CLI (file-based) and programmatic API (file-based or in-memory) workflows for both ONNX and PyTorch models.

CLI Usage (File-Based)
~~~~~~~~~~~~~~~~~~~~~~

**Example 1: ONNX Model**

The adapter takes a YAML file as input. You can include multiple passes in the same YAML file.

.. code-block:: yaml

    input_model_config:
      input_model_path: /path/to/input_model.onnx
    passes:
      onnx_convert_opset_version:
        target_opset_version: 21
      onnx_simplify:
        simplify: true
    output_model_path: /path/to/output_model.onnx

Run with:

.. code-block:: bash

   quark-cli shapeshifter config.yaml

**Example 2: PyTorch Model (File-Based)**

.. code-block:: yaml

    input_model_config:
      model_type: pytorch  # Discriminator field (required in YAML)
      input_model_path: /path/to/model.pt
      weights_only: false  # Required for loading nn.Module objects
      map_location: cpu
    passes:
      pytorch_remove_dropout: {}
      pytorch_trace_model:
        input_shapes:
          input: [1, 3, 224, 224]
        input_dtypes:
          input: float32
    output_model_path: /path/to/output.pt

Run with:

.. code-block:: bash

   quark-cli shapeshifter config.yaml

**Model Configuration Classes**

Shapeshifter uses an inheritance-based configuration system:

.. code-block:: python

    # Base class
    class ModelConfig:
        input_model_path: Path

    # ONNX-specific
    class ONNXModelConfig(ModelConfig):
        model_type: Literal["onnx"] = "onnx"

    # PyTorch-specific
    class PytorchModelConfig(ModelConfig):
        model_type: Literal["pytorch"] = "pytorch"
        weights_only: bool = False
        map_location: str = "cpu"

**Configuration Selection:**

* Use ``ONNXModelConfig`` for ONNX models (``.onnx`` files)
* Use ``PytorchModelConfig`` for PyTorch models (``.pt``, ``.pth``, ``.bin`` files)

**In YAML:** The ``model_type`` field acts as a Pydantic discriminator. You must specify it explicitly:

.. code-block:: yaml

    # For ONNX models
    input_model_config:
      model_type: onnx
      input_model_path: model.onnx

    # For PyTorch models
    input_model_config:
      model_type: pytorch
      input_model_path: model.pt
      weights_only: false

**In Python:** Use the specific config class directly (no model_type needed):

.. code-block:: python

    # ONNX
    config = ONNXModelConfig(input_model_path=Path("model.onnx"))

    # PyTorch
    config = PytorchModelConfig(
        input_model_path=Path("model.pt"),
        weights_only=False
    )

Programmatic API Usage
~~~~~~~~~~~~~~~~~~~~~~

**Example 1: ONNX Model (File-Based)**

.. code-block:: python

    from pathlib import Path
    from quark.shapeshifter import shapeshifter, RunConfig, ONNXModelConfig

    # Create ONNX model configuration
    model_config = ONNXModelConfig(input_model_path=Path("model.onnx"))

    run_config = RunConfig(
        input_model_config=model_config,
        passes={
            "onnx_simplify": {"simplify": True},
            "onnx_fold_batch_norm": {}
        },
        output_model_path="output.onnx"
    )

    # Run shapeshifter (saves to file)
    shapeshifter(run_config)

**Example 2: PyTorch Model (File-Based)**

.. code-block:: python

    from pathlib import Path
    from quark.shapeshifter import shapeshifter, RunConfig, PytorchModelConfig

    # Create PyTorch model configuration with load settings
    model_config = PytorchModelConfig(
        input_model_path=Path("model.pt"),
        weights_only=False,  # Required for loading nn.Module objects
        map_location="cuda:0"
    )

    run_config = RunConfig(
        input_model_config=model_config,
        passes={
            "pytorch_remove_dropout": {},
            "pytorch_trace_model": {
                "input_shapes": {"input": [1, 3, 224, 224]}
            }
        },
        output_model_path="output.pt"
    )

    # Run shapeshifter (saves to file)
    shapeshifter(run_config)

**Example 3: PyTorch Model (In-Memory)**

.. code-block:: python

    import torch.nn as nn
    from quark.shapeshifter import shapeshifter, RunConfig

    # Create or load model in memory
    model = MyModel()  # Your PyTorch model (nn.Module, function, or any callable)

    # Configure passes only (no input/output paths needed)
    run_config = RunConfig(
        passes={
            "pytorch_remove_dropout": {}
        }
    )

    # Run adapter on in-memory model
    optimized_model = shapeshifter(run_config, model=model)

    # Use optimized_model for inference or further processing

**Example 4: PyTorch Model (In-Memory with Optional Save)**

.. code-block:: python

    import torch.nn as nn
    from quark.shapeshifter import shapeshifter
    from quark.shapeshifter.run_config import RunConfig

    model = MyModel()

    # Configure with output path - will return model AND save to file
    run_config = RunConfig(
        passes={"pytorch_remove_dropout": {}},
        output_model_path="optimized_model.pt"
    )

    # Returns transformed model (also saves to file)
    optimized_model = shapeshifter(run_config, model=model)

Adding New Passes
-----------------

The `Shapeshifter` uses an automatic pass registration system that discovers and registers both ONNX and PyTorch passes dynamically. All passes (ONNX and PyTorch) are stored in a unified registry.

**Core Passes**

To add a new core pass, create a new Python file in the ``quark/shapeshifter/passes/`` directory following the conventions below.

**Community Passes**

Community-contributed passes can be added to the ``quark/contrib/shapeshifter_community_passes/`` directory. Community passes use the same registration system and are automatically discovered and registered to the global registry.

To contribute a community pass, follow the same implementation requirements as core passes (see below), but place the file in ``quark/contrib/shapeshifter_community_passes/`` instead of ``quark/shapeshifter/passes/``. Add your tests to ``quark/contrib/shapeshifter_community_passes/tests/`` and document your pass in :doc:`quark_shapeshifter_torch_passes` or :doc:`quark_shapeshifter_onnx_passes`.

Naming Conventions
~~~~~~~~~~~~~~~~~~

**File Naming:**
The filename (without the ``.py`` extension) becomes the pass name in the registry and is used in YAML configuration files.

* **ONNX passes:** Use ``onnx_`` prefix. Example: ``onnx_convert_bn_to_conv.py`` → Pass name: ``onnx_convert_bn_to_conv``
* **PyTorch passes:** Use ``pytorch_`` prefix. Example: ``pytorch_remove_dropout.py`` → Pass name: ``pytorch_remove_dropout``
* This creates a direct mapping between the implementation file and the registry name, making it easy to locate passes.
* **Important:** Files starting with ``_`` (underscore) are automatically skipped during discovery. This follows Python convention for private/internal modules. For example, ``_helper.py`` or ``__init__.py`` will not be processed as pass modules.

**Class Naming:**
Pass classes must end with ``Pass`` (e.g., ``ONNXConvertBNToConvPass``, ``PytorchRemoveDropoutPass``).

**Important:** Only classes ending with ``Pass`` are automatically discovered and included in the module's ``__all__`` list.

Pass Implementation Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**For ONNX Passes:**

1. **Subclass ONNXPass:** Your pass class must inherit from ``ONNXPass``.

2. **Use the @register_pass decorator:** Apply the decorator directly to your class (no parameters needed). The decorator automatically extracts the pass name from the filename.

3. **Implement required methods:**
   * ``_default_config()``: Return a dictionary of ``PassConfigParam`` objects defining the pass configuration parameters.
   * ``_run_for_config(model, config)``: Implement the actual pass logic that transforms the ONNX model. Takes ``onnx.ModelProto`` and returns ``onnx.ModelProto``.

**For PyTorch Passes:**

1. **Subclass PytorchPass:** Your pass class must inherit from ``PytorchPass``.

2. **Use the @register_pass decorator:** Apply the same decorator directly to your class.

3. **Implement required methods:**
   * ``_default_config()``: Return a dictionary of ``PassConfigParam`` objects defining the pass configuration parameters.
   * ``_run_for_config(model, config)``: Implement the actual pass logic that transforms the PyTorch model. Takes any ``Callable`` (e.g., ``torch.nn.Module``) and returns a ``Callable``.

Example Implementation
~~~~~~~~~~~~~~~~~~~~~~

**Example 1: ONNX Pass**

Here's a minimal example of a new ONNX pass:

.. code-block:: python

    #
    # Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
    # SPDX-License-Identifier: MIT
    #

    import onnx
    from onnx import ModelProto

    from quark.shapeshifter.pass_base import ONNXPass, register_pass
    from quark.shapeshifter.pass_config import PassConfigParam
    from quark.common.utils.log import ScreenLogger

    logger = ScreenLogger(__name__)


    @register_pass
    class ONNXMyCustomPass(ONNXPass):
        """Description of what this pass does."""

        def _default_config(self) -> dict[str, PassConfigParam]:
            """Return the default configuration for the pass."""
            config = {
                "enable_feature": PassConfigParam(
                    type_=bool,
                    default_value=True,
                    required=True,
                    description="Whether to enable the custom feature.",
                )
            }
            config.update(self.config)
            return config

        def _run_for_config(self, model: ModelProto, config: dict[str, PassConfigParam]) -> ModelProto:
            """Run the pass logic using the provided configuration."""
            if config.get("enable_feature", {}).get("value", False):
                # Implement your pass logic here
                logger.info("Applying custom pass transformation")
                # ... transform the model ...
            return model

**File Location:** ``quark/shapeshifter/passes/onnx_my_custom_pass.py``

**Usage in YAML:**

.. code-block:: yaml

    passes:
      onnx_my_custom_pass:
        enable_feature: true

**Example 2: PyTorch Pass**

Here's a minimal example of a new PyTorch pass:

.. code-block:: python

    #
    # Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
    # SPDX-License-Identifier: MIT
    #

    from typing import Any
    import torch.nn as nn

    from quark.shapeshifter.pass_base import PytorchPass, register_pass
    from quark.shapeshifter.pass_config import PassConfigParam
    from quark.common.utils.log import ScreenLogger

    logger = ScreenLogger(__name__)


    @register_pass
    class PytorchMyCustomPass(PytorchPass):
        """Description of what this PyTorch pass does."""

        def _default_config(self) -> dict[str, PassConfigParam]:
            """Return the default configuration for the pass."""
            config = {
                "enable_feature": PassConfigParam(
                    type_=bool,
                    default_value=True,
                    required=True,
                    description="Whether to enable the custom feature.",
                )
            }
            config.update(self.config)
            return config

        def _run_for_config(self, model: Any, config: dict[str, Any]) -> Any:
            """Run the pass logic using the provided configuration."""
            if config.get("enable_feature", True):
                logger.info("Applying custom PyTorch transformation")
                # Example: Modify model structure
                # model is any callable (nn.Module, function, etc.)
                # ... transform the model ...
            return model

**File Location:** ``quark/shapeshifter/passes/pytorch_my_custom_pass.py``

**Usage in YAML:**

.. code-block:: yaml

    input_model_config:
      model_type: pytorch  # Discriminator (required in YAML)
      input_model_path: model.pt
      weights_only: false
    passes:
      pytorch_my_custom_pass:
        enable_feature: true
    output_model_path: output.pt

Automatic Registration
~~~~~~~~~~~~~~~~~~~~~~

The pass registration system automatically:

* Discovers all ``.py`` files in the ``quark/shapeshifter/passes/`` directory
* Imports each module (triggering the ``@register_pass`` decorator)
* Registers **both ONNX and PyTorch passes** in a unified registry using the filename (without ``.py``) as the registry key
* Differentiates pass types by inheritance (``ONNXPass`` vs ``PytorchPass``)
* Makes pass classes available through the ``quark.shapeshifter.passes`` module

**No manual registration required:** Simply create your pass file with the ``@register_pass`` decorator, and it will be automatically discovered and registered when the passes module is imported.

**Type Safety:** The Engine automatically validates that all configured passes match the model type (ONNX or PyTorch). Mixing ONNX and PyTorch passes in a single workflow is not allowed and will raise a ``ValueError``.


Important Notes
~~~~~~~~~~~~~~~

**For ONNX Models:**

* Set ``SkipPreprocess`` to ``True`` in quantization config if Shapeshifter is used before quantization. See :doc:`Full List of Quantization Config Features <onnx/appendix_full_quant_config_features>` for details.

**For PyTorch Models:**

* PyTorch models can be any Python callable (``torch.nn.Module``, functions, custom callables)
* Use ``PytorchModelConfig`` for file-based loading
* Configuration fields:

  * ``weights_only`` (bool, default=False): If ``True``, only load state dict (more secure). If ``False``, load full model object (required for nn.Module). Note: PyTorch 2.6+ defaults to ``True`` for security.
  * ``map_location`` (str, default="cpu"): Device to load model on (``"cpu"``, ``"cuda"``, ``"cuda:0"``, etc.)

**Pass Validation:**

* All passes in a workflow must be the same type (all ONNX or all PyTorch)
* Mixing ONNX and PyTorch passes raises a ``ValueError``
* Model type is automatically detected from the first pass's inheritance (``ONNXPass`` or ``PytorchPass``)

Architecture and Design (For Developers)
----------------------------------------

This section provides detailed information about the Shapeshifter's internal architecture for developers who want to contribute or extend the system.

Conceptual Overview
~~~~~~~~~~~~~~~~~~~

The Shapeshifter is a **pass-based transformation framework** that applies sequential transformations to models:

1. **Input**: A model (ONNX graph or PyTorch callable)
2. **Processing**: A sequence of transformation passes
3. **Output**: An optimized/transformed model

Each **pass** is a self-contained transformation unit that:

* Takes a model as input
* Applies one specific transformation
* Returns the modified model

Passes are **composable** - they can be chained together, with each pass operating on the output of the previous one.

Core Components
~~~~~~~~~~~~~~~

**1. Configuration Classes (Type System)**

Configuration uses inheritance for type safety:

.. code-block:: python

    # Base class
    class ModelConfig(BaseModel):
        input_model_path: Path

    # ONNX-specific
    class ONNXModelConfig(ModelConfig):
        model_type: Literal["onnx"] = "onnx"

    # PyTorch-specific
    class PytorchModelConfig(ModelConfig):
        model_type: Literal["pytorch"] = "pytorch"
        weights_only: bool = False
        map_location: str = "cpu"

    # Top-level config
    class RunConfig(BaseModel):
        input_model_config: Union[ONNXModelConfig, PytorchModelConfig]
        passes: dict[str, dict[str, Any]]
        output_model_path: Optional[str]

**Design Benefits:**

* **Type detection via isinstance()**: No enum needed
* **Flat structure**: All fields at same level (no nested ``input_model_config``)
* **Pydantic discriminator**: Automatic deserialization from YAML using ``model_type`` field
* **IDE-friendly**: Autocomplete and type checking work correctly

**2. Engine (Orchestrator)**

The ``Engine`` class (``quark/shapeshifter/engine.py``) orchestrates the entire transformation pipeline:

* **Responsibilities:**

  * Load passes from the unified registry
  * Detect model type from config inheritance or pass inheritance
  * Validate all passes match the model type
  * Load models (from file or use in-memory)
  * Execute passes sequentially
  * Save outputs (to file or return in-memory)

* **Key Methods:**

  * ``initialize()``: Register all passes from ``REGISTRY``
  * ``_detect_model_type_from_config()``: Check config type (``isinstance()``)
  * ``_detect_model_type_from_passes()``: Fallback - check first pass inheritance
  * ``_validate_all_passes_match_type()``: Ensure no mixing of ONNX/PyTorch passes
  * ``run(float_model=None)``: Main execution method

* **Config Handling:**

  * Accepts ``RunConfig | dict[str, Any]``
  * Converts dict to RunConfig if needed (CLI path)
  * Keeps Pydantic objects throughout (no ``model_dump()``)

**3. Pass Base Classes**

``ONNXPass`` and ``PytorchPass`` (``quark/shapeshifter/pass_base.py``, ``onnx_pass.py``, ``pytorch_pass.py``) define the pass interface:

* **Template Method Pattern:**

  .. code-block:: python

      def run(model):
          if not initialized:
              self._initialize()    # Optional setup hook
          return self._run_for_config(model, config)  # Abstract - must implement

* **Required Methods:**

  * ``_default_config()``: Return configuration schema (abstract)
  * ``_run_for_config(model, config)``: Core transformation logic (abstract)

* **Optional Hooks:**

  * ``_initialize()``: Custom initialization
  * ``validate_config()``: Configuration validation

**4. Unified Registry**

The ``REGISTRY`` dictionary (global in ``pass_base.py``) maps pass names to pass classes:

.. code-block:: python

    REGISTRY = {
        "onnx_simplify": ONNXSimplifyPass,
        "onnx_fold_batch_norm": ONNXFoldBatchNormPass,
        "pytorch_remove_dropout": PytorchRemoveDropoutPass,
        "pytorch_trace_model": PytorchTraceModelPass,
        # ... all passes
    }

* **Populated by**: ``@register_pass`` decorator during module import
* **Single registry** for both ONNX and PyTorch passes
* **Type differentiation**: By inheritance (``ONNXPass`` vs ``PytorchPass``)

**5. Auto-Registration System**

The ``@register_pass`` decorator automatically registers passes:

.. code-block:: python

    @register_pass  # Extracts "pytorch_remove_dropout" from filename
    class PytorchRemoveDropoutPass(PytorchPass):
        ...

* **Mechanism:**

  1. Decorator inspects caller's ``__file__`` attribute
  2. Extracts filename without ``.py`` extension
  3. Validates inheritance (``ONNXPass`` or ``PytorchPass``)
  4. Adds to ``REGISTRY[filename] = cls``

* **Discovery:** ``quark/shapeshifter/passes/__init__.py`` uses ``pkgutil.iter_modules()`` to import all ``.py`` files (except those starting with ``_``)

Data Flow and Execution Sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Scenario 1: File-Based ONNX Model (CLI)**

.. code-block:: text

    User: quark-cli shapeshifter config.yaml
      │
      ├─ 1. CLI loads YAML → dict
      ├─ 2. CLI calls shapeshifter(RunConfig)
      ├─ 3. shapeshifter() creates Engine
      ├─ 4. Engine.initialize()
      │    └─ Loads all passes from REGISTRY
      ├─ 5. Engine.run(float_model=None)
      │    │
      │    ├─ 5a. Detect model type from first pass
      │    │    └─ issubclass(PassClass, ONNXPass)? → ONNX
      │    │
      │    ├─ 5b. Validate all passes are ONNX passes
      │    │
      │    ├─ 5c. Load model: onnx.load(path)
      │    │
      │    ├─ 5d. Execute passes sequentially
      │    │    └─ For each pass:
      │    │        └─ model = pass._run_for_config(model, config)
      │    │
      │    └─ 5e. Save: onnx.save(model, output_path)
      │
      └─ 6. Return None

**Scenario 2: In-Memory PyTorch Model (Programmatic API)**

.. code-block:: text

    User Code: shapeshifter(RunConfig, model=my_model)
      │
      ├─ 1. shapeshifter() creates Engine
      ├─ 2. Engine.initialize()
      ├─ 3. Engine.run(float_model=my_model)
      │    │
      │    ├─ 3a. Detect model type from passes
      │    │    └─ issubclass(PassClass, PytorchPass)? → PYTORCH
      │    │
      │    ├─ 3b. Validate all passes are PyTorch passes
      │    │
      │    ├─ 3c. Skip loading (model provided)
      │    │
      │    ├─ 3d. Execute passes
      │    │    └─ For each pass:
      │    │        └─ model = pass._run_for_config(model, config)
      │    │
      │    └─ 3e. Return transformed model
      │
      └─ 4. User receives optimized model

Model Type Detection
~~~~~~~~~~~~~~~~~~~~

The Engine determines whether to use ONNX or PyTorch workflow using a two-step process:

**Step 1: Check input_model_config type (preferred)**

.. code-block:: python

    # In Engine._detect_model_type_from_config()
    if isinstance(self._config.input_model_config, PytorchModelConfig):
        return PytorchModelConfig  # ✓ PyTorch workflow
    elif isinstance(self._config.input_model_config, ONNXModelConfig):
        return ONNXModelConfig  # ✓ ONNX workflow

**Step 2: Fallback to pass inheritance (for in-memory workflows)**

.. code-block:: python

    # In Engine._detect_model_type_from_passes()
    first_pass_cls = registry["pytorch_remove_dropout"]

    if issubclass(first_pass_cls, ONNXPass):
        return ONNXModelConfig
    elif issubclass(first_pass_cls, PytorchPass):
        return PytorchModelConfig  # ✓ Returns config class, not enum

**Key Design:**
- No ``ModelType`` enum needed - uses config class types directly
- Type detection via ``isinstance()`` checks (Pythonic)
- Config class (``ONNXModelConfig`` vs ``PytorchModelConfig``) carries type information

**Validation:** All subsequent passes must match the detected type, otherwise ``ValueError`` is raised.

Design Patterns
~~~~~~~~~~~~~~~

**1. Registry Pattern**

* **Where**: ``REGISTRY`` dict + ``@register_pass`` decorator
* **Purpose**: Decouple pass discovery from Engine
* **Benefit**: Passes auto-register on import; no manual registration

**2. Template Method Pattern**

* **Where**: ``run()`` method in base classes
* **Purpose**: Define algorithm skeleton, let subclasses fill details
* **Flow**: ``initialize() → validate() → _run_for_config()`` (abstract)

**3. Strategy Pattern**

* **Where**: Pass selection and execution
* **Purpose**: Swap transformation algorithms at runtime
* **Mechanism**: Config specifies which passes to use

**4. Decorator Pattern**

* **Where**: ``@register_pass``
* **Purpose**: Add registration behavior to classes
* **Benefit**: Clean syntax, automatic registration

Key Design Decisions
~~~~~~~~~~~~~~~~~~~~

**Decision 1: Unified Registry**

* **Rationale**: Simplifies engine logic; type checking via inheritance
* **Trade-off**: Requires careful validation to prevent mixing ONNX/PyTorch passes
* **Implementation**: Single ``REGISTRY`` dict, differentiation by ``issubclass()`` check

**Decision 2: Inheritance-Based Configuration**

* **Rationale**: Type safety via polymorphism; no enums needed; flat config structure
* **Trade-off**: Requires discriminator field in YAML; more classes to understand
* **Implementation**: ``ONNXModelConfig`` and ``PytorchModelConfig`` inherit from ``ModelConfig``
* **Benefits**: ``isinstance()`` checks, IDE autocomplete, Pydantic validation throughout

**Decision 3: Model Type Detection Strategy**

* **Primary**: Check ``isinstance(config.input_model_config, PytorchModelConfig)``
* **Fallback**: Inspect first pass inheritance (for in-memory workflows without config)
* **Rationale**: Config type is authoritative source of truth when available
* **Trade-off**: Mixing passes still not allowed (validated separately)

**Decision 4: Pydantic Throughout (No Dict Conversion)**

* **Rationale**: Keep type information; enable validation; improve developer experience
* **Implementation**:
  * API: Pass ``RunConfig`` directly to Engine
  * Engine: Accept ``RunConfig | dict``, convert dict→RunConfig if needed (CLI path)
  * All access via object attributes: ``config.passes`` not ``config["passes"]``
* **Benefits**: Type safety, IDE autocomplete, Pydantic validation active

**Decision 5: File-Based + In-Memory Support**

* **Rationale**:

  * CLI needs file-based (can't pass Python objects in YAML)
  * Programmatic API benefits from in-memory (no I/O overhead)

* **Implementation**: ``run(float_model=None)`` - if ``None``, load from file

**Decision 6: Callable Type for PyTorch Models**

* **Rationale**: Maximum flexibility (``nn.Module``, functions, custom callables)
* **Trade-off**: Less type safety; runtime errors if not actually callable
* **Validation**: Check ``callable(model)`` when loading from file

Extension Points
~~~~~~~~~~~~~~~~

**Custom Model Loaders:**

To support new model formats (e.g., HuggingFace):

1. Extend ``PytorchModelConfig`` with new fields
2. Add loading logic in ``Engine._load_pytorch_model()``
3. Check new config field and call custom loader

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Pass Execution Order:**

* Passes execute sequentially in the order specified in config
* Each pass receives the output of the previous pass
* No parallel execution (intentional - ensures deterministic results)

**Memory Management:**

* ONNX models: Entire graph loaded into memory
* PyTorch models: Model weights loaded into specified device (``map_location``)
* In-memory mode: No I/O overhead between passes

**Optimization Tips:**

* Place fast passes first (fail fast on validation errors)
* Group related passes together (e.g., all BatchNorm folding)
* Use ``onnx_simplify`` early to reduce graph size for subsequent passes

Testing Strategy
~~~~~~~~~~~~~~~~

**Unit Tests:**

* Test each pass independently with minimal models
* Verify transformation correctness
* Check edge cases (empty models, unsupported ops, etc.)

**Integration Tests:**

* Test full workflows (multiple passes)
* Test both CLI and programmatic API
* Test file-based and in-memory modes
* Verify pass validation (mixing ONNX/PyTorch should fail)

**Example Test Pattern:**

.. code-block:: python

    @use_temporary_directory
    def test_pytorch_pass(self, tmpdir):
        model = SimpleModel()
        model_path = tmpdir / "model.pt"
        torch.save(model, model_path)

        config = {
            "input_model_config": {"input_model_path": str(model_path)},
            "passes": {"pytorch_remove_dropout": {}},
            "output_model_path": str(tmpdir / "output.pt")
        }

        cli(["shapeshifter", str(config_path)])

        output_model = torch.load(tmpdir / "output.pt")
        assert not has_dropout(output_model)

Troubleshooting
~~~~~~~~~~~~~~~

**Common Issues:**

1. **"Pass not registered" error:**

   * Ensure filename matches pass name in config
   * Check file doesn't start with ``_``
   * Verify ``@register_pass`` decorator is applied

2. **"Not a PyTorch/ONNX pass" error:**

   * You're mixing ONNX and PyTorch passes
   * All passes must match the model type
   * Check inheritance of each pass class

3. **"Model is not callable" error:**

   * PyTorch checkpoint doesn't contain a callable
   * Try setting ``weights_only: false`` in ``input_model_config``
   * Ensure you saved the full model, not just state dict

4. **Import errors during pass discovery:**

   * Check pass file for syntax errors
   * Ensure all dependencies are installed
   * Check logs for specific import error message

Further Reading
~~~~~~~~~~~~~~~

* **Source Code:**

  * ``quark/shapeshifter/engine.py`` - Main orchestration logic
  * ``quark/shapeshifter/passes/`` - All pass implementations
  * ``quark/shapeshifter/api.py`` - Programmatic API entry point

* **Related Documentation:**

  * :doc:`Full List of Quantization Config Features <onnx/appendix_full_quant_config_features>`
  * Quark PyTorch Quantization Guide
  * Quark ONNX Quantization Guide
