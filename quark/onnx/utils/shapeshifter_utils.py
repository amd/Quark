#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import os
from pathlib import Path
from typing import Any, NamedTuple

import onnx
from onnxruntime.quantization.onnx_model import ONNXModel

from quark.common.utils.log import ScreenLogger
from quark.onnx.quantization.quant_utils import (
    get_pre_defined_preprocess_config,
    model_size_exceeds,
)
from quark.onnx.utils.model_utils import save_onnx_model_with_external_data

logger = ScreenLogger(__name__)


def load_shapeshifter_pass_groups(spec: str) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    """Resolve a ``ShapeShifterYaml`` spec into its pass groups and preprocessed-model path.

    ``spec`` may be a path to a ``.yaml`` file using the two-group schema
    (``preprocess_passes:`` / ``postprocess_passes:``) or the name of a pre-defined
    template (e.g. ``"xint8"``). For backward compatibility with the legacy
    ``PreprocessYAML`` schema (and the pre-defined templates), a flat ``passes:``
    mapping is treated as the preprocessing group.

    Returns a ``(preprocess_passes, postprocess_passes, preprocessed_model_path)``
    tuple, where ``preprocessed_model_path`` is the optional ``preprocessed_model_path``
    field from the YAML (``None`` when not specified).
    """
    from quark.shapeshifter import LoadConfigFromFileOrDict

    if spec.endswith(".yaml"):
        config = LoadConfigFromFileOrDict(spec).data
    else:
        config = get_pre_defined_preprocess_config(spec)

    pre_passes = config.get("preprocess_passes")
    post_passes = config.get("postprocess_passes")
    if pre_passes is None and post_passes is None and "passes" in config:
        # Legacy flat schema / pre-defined template: everything is preprocessing.
        pre_passes = config["passes"]

    return pre_passes or {}, post_passes or {}, config.get("preprocessed_model_path")


def run_shapeshifter_stage(model: onnx.ModelProto, passes: dict[str, Any]) -> onnx.ModelProto:
    """Run a group of Shapeshifter passes on an in-memory ONNX model and return it.

    When ``passes`` is empty the model is returned unchanged. Passes are handed to
    the Shapeshifter ``Engine`` as a flat ``passes`` config; the model type is
    detected from the pass inheritance since the model is provided in-memory. The
    result is topologically sorted, since some passes (e.g. ``onnx_align_scale``)
    reconnect nodes and may leave the graph out of topological order.
    """
    if not passes:
        return model

    from quark.shapeshifter import Engine

    engine = Engine(config={"passes": passes})
    engine.initialize()
    model = engine.run(float_model=model)  # type: ignore
    onnx_model = ONNXModel(model)
    onnx_model.topological_sort()
    return onnx_model.model


class ShapeShifterPreprocessResult(NamedTuple):
    """Outcome of the ShapeShifter preprocessing stage.

    :param model: the float model after the preprocess passes (unchanged when no
        ``ShapeShifterYaml`` / ``PreprocessYAML`` spec was given).
    :param postprocess_passes: the ``postprocess_passes`` group, to be handed to
        :func:`run_shapeshifter_stage` after the quantized model is produced. Empty
        when there is no spec or the spec declares no postprocessing.
    :param include_cle: the possibly-updated ``include_cle`` flag; forced to False
        when CLE is declared as a ShapeShifter preprocess pass.
    :param skip_pre_process_graph_optimization: the possibly-updated flag; forced to
        True whenever a spec is in effect.
    """

    model: onnx.ModelProto
    postprocess_passes: dict[str, Any]
    include_cle: bool
    skip_pre_process_graph_optimization: bool


def apply_shapeshifter_preprocess(
    float_model: onnx.ModelProto,
    model_input: str | Path | onnx.ModelProto,
    extra_options: dict[str, Any],
    *,
    include_cle: bool,
    skip_pre_process_graph_optimization: bool,
    use_external_data_format: bool,
) -> ShapeShifterPreprocessResult:
    """Run the ShapeShifter preprocessing stage for ``quantize_static``.

    Resolves the ``ShapeShifterYaml`` extra option (falling back to the deprecated
    ``PreprocessYAML``), runs the preprocess pass group on ``float_model``, persists
    the preprocessed float model, and returns the postprocess pass group for the
    caller to apply after quantization.

    When neither option is set this is a no-op: the inputs are returned unchanged
    with an empty postprocess group.

    :param float_model: the float model to preprocess.
    :param model_input: the original ``quantize_static`` input, used to derive the
        default preprocessed-model path.
    :param extra_options: ``quantize_static`` extra_options dict; read-only here.
    :param include_cle: the ``include_cle`` flag from ``quantize_static``.
    :param skip_pre_process_graph_optimization: the ``SkipPreprocess`` derived flag.
    :param use_external_data_format: whether to save the preprocessed model with
        external data.
    """
    shapeshifter_yaml_spec = extra_options.get("ShapeShifterYaml")
    pre_process_yaml_path = extra_options.get("PreprocessYAML")
    if pre_process_yaml_path is not None:
        logger.warning(
            "The 'PreprocessYAML' option is deprecated and will be removed in a future "
            "release; use 'ShapeShifterYaml' instead, which also supports postprocessing "
            "passes via a 'postprocess_passes' group."
        )
        if shapeshifter_yaml_spec is not None:
            logger.warning(
                "Both 'ShapeShifterYaml' and 'PreprocessYAML' were provided; "
                "'ShapeShifterYaml' takes precedence and 'PreprocessYAML' is ignored."
            )
        else:
            shapeshifter_yaml_spec = pre_process_yaml_path

    if shapeshifter_yaml_spec is None:
        return ShapeShifterPreprocessResult(
            model=float_model,
            postprocess_passes={},
            include_cle=include_cle,
            skip_pre_process_graph_optimization=skip_pre_process_graph_optimization,
        )

    skip_pre_process_graph_optimization = True

    (
        shapeshifter_preprocess_passes,
        shapeshifter_postprocess_passes,
        shapeshifter_preprocessed_model_path,
    ) = load_shapeshifter_pass_groups(shapeshifter_yaml_spec)

    # Avoid running CLE twice: when it is declared as a ShapeShifter preprocess pass,
    # the ShapeShifter YAML is the single source of truth, so the legacy `include_cle`
    # flag (which defaults to True) is turned off to prevent a second application in
    # `apply_pre_process`.
    if "onnx_cross_layer_equalization" in shapeshifter_preprocess_passes and include_cle:
        logger.warning("CLE is declared in ShapeShifterYaml; ignoring include_cle to avoid running it twice.")
        include_cle = False

    float_model = run_shapeshifter_stage(float_model, shapeshifter_preprocess_passes)

    # Persist the float model after the preprocess passes have been applied. The
    # destination is the ShapeShifterYaml ``preprocessed_model_path`` field, or, when
    # that field is absent, ``<input_model_name>_preprocessed.onnx`` in the same
    # directory as the input model.
    if shapeshifter_preprocess_passes:
        preprocessed_model_path = shapeshifter_preprocessed_model_path
        if not preprocessed_model_path:
            if isinstance(model_input, str | Path):
                input_dir = os.path.dirname(str(model_input))
                input_stem = os.path.splitext(os.path.basename(str(model_input)))[0]
            else:
                input_dir = ""
                input_stem = "model"
            preprocessed_model_path = os.path.join(input_dir, f"{input_stem}_preprocessed.onnx")
        save_onnx_model_with_external_data(
            float_model,
            preprocessed_model_path,
            save_as_external_data=use_external_data_format or model_size_exceeds(float_model),
        )
        logger.info(f"Saved the preprocessed float model to '{preprocessed_model_path}'.")

    return ShapeShifterPreprocessResult(
        model=float_model,
        postprocess_passes=shapeshifter_postprocess_passes,
        include_cle=include_cle,
        skip_pre_process_graph_optimization=skip_pre_process_graph_optimization,
    )
