#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import pytest

import quark.torch.extensions.brevitas.algos as brevitas_algos
import quark.torch.extensions.brevitas.config as brevitas_config
import quark.torch.extensions.brevitas.verification as brevitas_verify
import quark.torch.quantization.config.type as quark_type


def test_valid_default_config():
    config = brevitas_config.Config(global_quant_config=brevitas_config.QLayerConfig())

    try:
        brevitas_verify.ConfigVerifier.verify_config(config)
    except ValueError as e:
        pytest.fail(f"Default config shouldn't raise exception: {e}")


def test_unsupported_backend():
    config = brevitas_config.Config(global_quant_config=brevitas_config.QLayerConfig(), backend=None)
    with pytest.raises(ValueError):
        brevitas_verify.ConfigVerifier.verify_config(config)


def test_missing_floating_point_parameters():
    # exponent and mantissa must be specified if float_quant selected
    weight_spec = brevitas_config.QTensorConfig(quant_type=brevitas_config.QuantType.float_quant)
    global_config = brevitas_config.QLayerConfig(weight=weight_spec)
    config = brevitas_config.Config(global_quant_config=global_config)

    with pytest.raises(ValueError):
        brevitas_verify.ConfigVerifier.verify_config(config)


def test_invalid_incorrect_algorithm_parameter():
    global_config = brevitas_config.QLayerConfig()
    config = brevitas_config.Config(
        global_quant_config=global_config,
        # laywise must be true if the backend type is layerwise
        pre_quant_opt_config=[brevitas_algos.ActivationEqualization(is_layerwise=False)],
        backend=brevitas_config.Backend.layerwise,
    )

    with pytest.raises(ValueError):
        brevitas_verify.ConfigVerifier.verify_config(config)


invalid_combinations = [
    [brevitas_algos.GPTQ(), brevitas_algos.GPFA2Q()],
    [brevitas_algos.GPTQ(), brevitas_algos.GPFQ()],
    [brevitas_algos.GPFQ(), brevitas_algos.GPFA2Q()],
    [brevitas_algos.GPTQ(), brevitas_algos.GPFQ(), brevitas_algos.GPFA2Q()],
]


@pytest.mark.parametrize("algo_combinations", invalid_combinations)
def test_invalid_algorithm_combinations(algo_combinations):
    global_config = brevitas_config.QLayerConfig()
    config = brevitas_config.Config(global_quant_config=global_config, algo_config=algo_combinations)

    with pytest.raises(ValueError):
        brevitas_verify.ConfigVerifier.verify_config(config)


valid_combinations = [
    [brevitas_algos.GPTQ()],
    [brevitas_algos.GPFQ()],
    [brevitas_algos.GPFA2Q()],
    [brevitas_algos.GPTQ(), brevitas_algos.BiasCorrection()],
]


@pytest.mark.parametrize("algo_combinations", valid_combinations)
def test_valid_algorithm_combinations(algo_combinations):
    global_config = brevitas_config.QLayerConfig(
        weight=brevitas_config.QTensorConfig(), input_tensors=brevitas_config.QTensorConfig()
    )
    config = brevitas_config.Config(global_quant_config=global_config, algo_config=algo_combinations)

    try:
        brevitas_verify.ConfigVerifier.verify_config(config)
    except ValueError as e:
        pytest.fail(f"Config shouldn't raise exception: {e}")


algos_needing_input_quant = [
    [brevitas_algos.GPFQ()],
    [brevitas_algos.GPFA2Q()],
]


@pytest.mark.parametrize("algo", algos_needing_input_quant)
def test_missing_input_quant(algo):
    global_config = brevitas_config.QLayerConfig(weight=brevitas_config.QTensorConfig(), input_tensors=None)
    config = brevitas_config.Config(global_quant_config=global_config, algo_config=algo)

    with pytest.raises(ValueError):
        brevitas_verify.ConfigVerifier.verify_config(config)


def test_invalid_asym_float_quant():
    weight_spec = brevitas_config.QTensorConfig(
        quant_type=brevitas_config.QuantType.float_quant,
        exponent_bit_width=4,
        mantissa_bit_width=3,
        # must be symmetric with floating point quantization
        symmetric=False,
    )
    global_config = brevitas_config.QLayerConfig(weight=weight_spec)
    config = brevitas_config.Config(global_quant_config=global_config)

    with pytest.raises(ValueError):
        brevitas_verify.ConfigVerifier.verify_config(config)


def test_invalid_bias_without_input_quant():
    bias_spec = brevitas_config.QTensorConfig()
    global_config = brevitas_config.QLayerConfig(bias=bias_spec)
    config = brevitas_config.Config(global_quant_config=global_config)

    with pytest.raises(ValueError):
        brevitas_verify.ConfigVerifier.verify_config(config)


def test_valid_input_bias_quant():
    input_spec = brevitas_config.QTensorConfig()
    bias_spec = brevitas_config.QTensorConfig()
    global_config = brevitas_config.QLayerConfig(input_tensors=input_spec, bias=bias_spec)
    config = brevitas_config.Config(global_quant_config=global_config)

    try:
        brevitas_verify.ConfigVerifier.verify_config(config)
    except ValueError as e:
        pytest.fail(f"Config shouldn't raise exception: {e}")


def test_invalid_bias_quant_type():
    input_spec = brevitas_config.QTensorConfig()
    bias_spec = brevitas_config.QTensorConfig(
        quant_type=brevitas_config.QuantType.float_quant, exponent_bit_width=4, mantissa_bit_width=3
    )
    global_config = brevitas_config.QLayerConfig(input_tensors=input_spec, bias=bias_spec)
    config = brevitas_config.Config(global_quant_config=global_config)

    with pytest.raises(ValueError):
        brevitas_verify.ConfigVerifier.verify_config(config)


def test_verify_bias_spec_no_error():
    # these arguments are ignored but generated a series of warning dialogs to tell the user
    spec = brevitas_config.QTensorConfig(
        qscheme=quark_type.QSchemeType.per_channel,
        symmetric=False,
        scale_type=quark_type.ScaleType.pof2,
        param_type=brevitas_config.ParamType.mse,
        exponent_bit_width=4,
        mantissa_bit_width=3,
    )
    try:
        brevitas_verify.ConfigVerifier._verify_bias_quant_spec(spec)
    except ValueError as e:
        pytest.fail(f"Spec shouldn't raise exception: {e}")


def test_verify_preprocess_wrong_order_no_error():
    global_config = brevitas_config.QLayerConfig()
    config = brevitas_config.Config(
        global_quant_config=global_config,
        pre_quant_opt_config=[brevitas_algos.PreQuantOptConfig(), brevitas_algos.Preprocess()],
    )
    try:
        brevitas_verify.ConfigVerifier.verify_config(config)
    except ValueError as e:
        pytest.fail(f"Config shouldn't raise exception: {e}")
