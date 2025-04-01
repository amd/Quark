#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from quark.torch.quantization.config.config import Int4PerGroupSpec, Config, AWQConfig, PreQuantOptConfig, QuantizationConfig

INT4_PER_GROUP_SYM_SPEC = Int4PerGroupSpec(symmetric=True,
                                           scale_type="float",
                                           round_method="half_even",
                                           ch_axis=1,
                                           is_dynamic=False,
                                           group_size=128).to_quantization_spec()

def test_set_group_size():
    # Create an instance of QuantizationSpec

    # Set group size
    new_group_size = 8
    INT4_PER_GROUP_SYM_SPEC.set_group_size(new_group_size)

    # Assert the group size was set correctly
    assert INT4_PER_GROUP_SYM_SPEC.group_size == new_group_size, "The group size should be updated to the new value"

    # Test with group_size = -1 (valid case)
    INT4_PER_GROUP_SYM_SPEC.set_group_size(-1)
    assert INT4_PER_GROUP_SYM_SPEC.group_size == -1, "The group size should be set to -1"

    # Test with group_size = 0.1 (invalid case)
    try:
        INT4_PER_GROUP_SYM_SPEC.set_group_size(0.1)
    except AssertionError as e:
        assert str(e) == "Group size must be a positive integer or -1 (which means group size equals to dimension size).", "Expected AssertionError for invalid group size"

def test_set_algo_config():
    # Initialize a Config object
    config = Config(global_quant_config=QuantizationConfig(weight=INT4_PER_GROUP_SYM_SPEC))

    # Create an AlgoConfig object to be set
    new_algo_config = AWQConfig()  # Assuming AlgoConfig has no required arguments

    # Set the algo_config
    config.set_algo_config(new_algo_config)

    # Assert algo_config is correctly updated
    assert config.algo_config == new_algo_config, "algo_config should be updated with the new AlgoConfig object"
    assert isinstance(config.algo_config, AWQConfig), "algo_config should be an instance of AlgoConfig"

def test_add_pre_optimization_config():
    # Initialize a Config object
    config = Config(global_quant_config=QuantizationConfig(weight=INT4_PER_GROUP_SYM_SPEC))

    # Create a PreQuantOptConfig object to be added
    new_pre_opt_config = PreQuantOptConfig()  # Assuming PreQuantOptConfig has no required arguments

    # Add the PreQuantOptConfig to the list
    config.add_pre_optimization_config(new_pre_opt_config)

    # Assert the config contains the added PreQuantOptConfig
    assert new_pre_opt_config in config.pre_quant_opt_config, "pre_quant_opt_config should include the new PreQuantOptConfig object"
    assert len(config.pre_quant_opt_config) == 1, "There should be one PreQuantOptConfig object in the list"
    assert isinstance(config.pre_quant_opt_config[0], PreQuantOptConfig), "Items in pre_quant_opt_config should be instances of PreQuantOptConfig"
