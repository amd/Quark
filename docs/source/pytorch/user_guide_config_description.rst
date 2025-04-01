Configuring PyTorch Quantization
================================

This topic describes the steps on how to set the quantization configuration in Quark for PyTorch.

Configuration of quantization in ``Quark for PyTorch`` is set by Python ``dataclass`` because it is rigorous and can help users avoid typos.
We provide a class ``Config`` in ``quark.torch.quantization.config.config`` for configuration. There are several steps to set up the configuration.

- Step 1: Configure ``QuantizationSpec`` for torch.Tensors. Specify attributes such as dtype, observer_cls, etc.
- Step 2: Establish ``QuantizationConfig`` for nn.Module. Define the QuantizationSpec of input_tensors, output_tensors, weight, and bias.
- Step 3: [Optional] Set ``AlgoConfig`` for the model.
- Step 4: Set up the overall ``Config`` for the model. This includes:


.. toctree::
  :hidden:
  :maxdepth: 1

  Calibration Methods <calibration_methods.rst>
  Calibration Datasets <calibration_datasets.rst>
  Quantization Strategies <quantization_strategies.rst>
  Quantization Schemes <quantization_schemes.rst>
  Quantization Symmetry <quantization_symmetry.rst>

Step 1: Configuring ``QuantizationSpec`` for torch.Tensors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``QuantizationSpec`` aims to describe the quantization specification for each tensor, including dtype, observer_cls, qscheme, is_dynamic, symmetric, etc. For example:

.. code:: python

   from quark.torch.quantization.config.config import QuantizationSpec
   from quark.torch.quantization.config.type import Dtype, QSchemeType, ScaleType, RoundType
   from quark.torch.quantization.observer.observer import PlaceholderObserver, PerTensorMinMaxObserver, PerGroupMinMaxObserver

   BFLOAT16_SPEC = QuantizationSpec(dtype=Dtype.bfloat16, observer_cls=PlaceholderObserver)

   FP8_PER_TENSOR_SPEC = QuantizationSpec(dtype=Dtype.fp8_e4m3,
                                          qscheme=QSchemeType.per_tensor,
                                          observer_cls=PerTensorMinMaxObserver,
                                          is_dynamic=False)

   INT8_PER_TENSOR_SPEC = Int8PerTensorSpec(observer_method="min_max",
                                           symmetric=True,
                                           scale_type=ScaleType.float,
                                           round_method=RoundType.half_even,
                                           is_dynamic=False)

   UINT4_PER_GROUP_ASYM_SPEC = QuantizationSpec(dtype=Dtype.uint4,
                                                observer_cls=PerGroupMinMaxObserver,
                                                symmetric=False,
                                                scale_type=ScaleType.float,
                                                round_method=RoundType.half_even,
                                                qscheme=QSchemeType.per_group,
                                                ch_axis=1,
                                                is_dynamic=False,
                                                group_size=128)

For parameter explanation:

.. list-table:: Parameter Explanation
   :widths: 20 40 20 20
   :header-rows: 1

   * - Name
     - Description
     - Class Type
     - Option
   * - ``dtype``
     - The data type for quantization.
     - ``Dtype``
     - ``Dtype.int8``, ``Dtype.uint8``, ``Dtype.int4``, ``Dtype.uint4``, ``Dtype.int2``, ``Dtype.bfloat16``, ``Dtype.float16``, ``Dtype.fp8_e5m2``, ``Dtype.fp8_e4m3``, ``Dtype.fp6_e3m2``, ``Dtype.fp6_e2m3``, ``Dtype.fp4``, ``Dtype.mx``, ``Dtype.mx6``, ``Dtype.mx9``
   * - ``observer_cls``
     - The class of observer to be used for determining quantization parameters.
     - ``Optional[ObserverBase]``
     - ``PlaceholderObserver``, ``PerTensorMinMaxObserver``, ``PerChannelMinMaxObserver``, ``PerGroupMinMaxObserver``, ``PerBlockMXObserver``, ``PerTensorPercentileObserver``, ``PerTensorMSEObserver``
   * - ``symmetric``
     - Specifies whether dynamic or static quantization should be used.
     - ``Optional[bool]``
     - True, False, None
   * - ``scale_type``
     - The scale type to be used for quantization
     - ``Optional[ScaleType]``
     - ``ScaleType.float``, ``ScaleType.pof2``
   * - ``round_method``
     - The rounding method during quantization.
     - ``Optional[RoundType]``
     - ``RoundType.round``, ``RoundType.floor``, ``RoundType.half_even``
   * - ``qscheme``
     - The quantization scheme to use.
     - ``Optional[QSchemeType]``
     - ``QSchemeType.per_tensor``, ``QSchemeType.per_channel``, ``QSchemeType.per_group``
   * - ``ch_axis``
     - The channel axis for per-channel quantization.
     - ``Optional[int]``
     - int, None
   * - ``is_dynamic``
     - Specifies whether dynamic or static quantization should be used.
     - ``Optional[bool]``
     - True, False, None
   * - ``group_size``
     - The size of the group for per-group quantization.
     - ``Optional[int]``
     - int, None

Step 2: Establishing ``QuantizationConfig`` for nn.Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``QuantizationConfig`` is used to describe the global, layer-type-wise, or layer-wise quantization information for each ``nn.Module``, such as ``nn.Linear``. For example,

.. code:: python

   from quark.torch.quantization.config.config import QuantizationConfig

   W_FP8_A_FP8_PER_TENSOR_CONFIG = QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC,
                                                      weight=FP8_PER_TENSOR_SPEC)

   W_INT8_A_INT8_PER_TENSOR_CONFIG = QuantizationConfig(input_tensors=INT8_PER_TENSOR_SPEC,
                                                        weight=INT8_PER_TENSOR_SPEC)

   W_UINT4_PER_GROUP_CONFIG = QuantizationConfig(weight=UINT4_PER_GROUP_ASYM_SPEC)

For parameter explanation:

.. list-table:: Parameter Explanation
   :widths: 20 20 20
   :header-rows: 1

   * - Name
     - Class Type
     - Default
   * - ``input_tensors``
     - ``Optional[QuantizationSpec]``
     - None
   * - ``output_tensors``
     - ``Optional[QuantizationSpec]``
     - None
   * - ``weight``
     - ``Optional[QuantizationSpec]``
     - None
   * - ``bias``
     - ``Optional[QuantizationSpec]``
     - None

Step 3: [Optional] Setting ``AlgoConfig`` for the model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If users want to use Quark's advanced algorithms such as AWQ, they should set up the configuration for them.

Users should possess a thorough understanding of the methods and hyper-parameters associated with the algorithms prior to configuring them!
Algorithms only support some ``QuantizationSpec``, please make sure before running.

Here we use the algorithms configuration of Llama2-7b as the example:

.. code:: python

   from quark.torch.algorithm.awq.awq import AwqProcessor
   from quark.torch.algorithm.awq.smooth import SmoothQuantProcessor
   from quark.torch.algorithm.gptq.gptq import GptqProcessor
   from quark.torch.quantization.config.config import AWQConfig, SmoothQuantConfig, GPTQConfig

   ALGORITHM_CONFIG=AWQConfig(
     scaling_layers=[
       {'prev_op': 'input_layernorm', 'layers': ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'], 'inp': 'self_attn.q_proj', 'module2inspect': 'self_attn'}, 
       {'prev_op': 'self_attn.v_proj', 'layers': ['self_attn.o_proj'], 'inp': 'self_attn.o_proj'}, 
       {'prev_op': 'post_attention_layernorm', 'layers': ['mlp.gate_proj', 'mlp.up_proj'], 'inp': 'mlp.gate_proj', 'module2inspect': 'mlp', 'help': 'linear 1'}, 
       {'prev_op': 'mlp.up_proj', 'layers': ['mlp.down_proj'], 'inp': 'mlp.down_proj',  'help': 'linear 2'}], 
     model_decoder_layers='model.layers')

   ALGORITHM_CONFIG=SmoothQuantConfig(
     alpha=0.5,
     scale_clamp_min=0.001,
     scaling_layers=[
       {'prev_op': 'input_layernorm', 'layers': ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'], 'inp': 'self_attn.q_proj', 'module2inspect': 'self_attn'}, 
       {'prev_op': 'self_attn.v_proj', 'layers': ['self_attn.o_proj'], 'inp': 'self_attn.o_proj'}, 
       {'prev_op': 'post_attention_layernorm', 'layers': ['mlp.gate_proj', 'mlp.up_proj'], 'inp': 'mlp.gate_proj', 'module2inspect': 'mlp', 'help': 'linear 1'}, 
       {'prev_op': 'mlp.up_proj', 'layers': ['mlp.down_proj'], 'inp': 'mlp.down_proj',   'help': 'linear 2'}], 
     model_decoder_layers='model.layers')

   ALGORITHM_CONFIG = GPTQConfig(
       damp_percent=0.01,
       desc_act=True,
       static_groups=True,
       true_sequential=True,
       inside_layer_modules=['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj', 'self_attn.o_proj', 'mlp.up_proj', 'mlp.gate_proj', 'mlp.down_proj'],
       model_decoder_layers='model.layers'
   )

For AWQ, Quark for PyTorch only supports ``AWQ`` with quantization data type as ``uint4/int4`` and ``per group``, running on ``Linux`` with the ``GPU mode`` for now. Parameter explanation:

.. list-table:: Parameter Explanation
   :widths: 20 20 20
   :header-rows: 1

   * - Name
     - Class Type
     - Default
   * - ``scaling_layers``
     - ``Optional[List[Dict[str, str]]]``
     - None
   * - ``model_decoder_layers``
     - ``Optional[str]``
     - None

For SmoothQuant parameter explanation:

.. list-table:: Parameter Explanation
   :widths: 20 20 20
   :header-rows: 1

   * - Name
     - Class Type
     - Default
   * - ``alpha``
     - float
     - 1
   * - ``scale_clamp_min``
     - float
     - 1e-3
   * - ``scaling_layers``
     - ``Optional[List[Dict[str, str]]]``
     - None
   * - ``model_decoder_layers``
     - ``Optional[str]``
     - None


.. list-table:: Parameter Explanation
   :widths: 20 20 20
   :header-rows: 1

   * - Name
     - Class Type
     - Default
   * - ``damp_percent``
     - float
     - 0.01
   * - ``desc_act``
     - bool
     - True
   * - ``static_groups``
     - bool
     - True
   * - ``true_sequential``
     - bool
     - True
   * - ``inside_layer_modules``
     - ``Optional[List[str]]``
     - None
   * - ``model_decoder_layers``
     - ``Optional[str]``
     - None

More details about SmoothQuant parameters are available in :doc:`Activation/weight smoothing (SmoothQuant) documentation <smoothquant>`.

For GPTQ, Quark for PyTorch only supports ``GPTQ`` with quantization
data type as ``uint4/int4`` and ``per group``, running on ``Linux`` with
the ``GPU mode`` for now. parameter explanation:

+------------------------+----------------------------------+---------+
| Name                   | Class Type                       | Default |
+------------------------+----------------------------------+---------+
| ``damp_percent``       | float                            | 0.01    |
+------------------------+----------------------------------+---------+
| ``desc_act``           | bool                             | True    |
+------------------------+----------------------------------+---------+
| ``static_groups``      | bool                             | True    |
+------------------------+----------------------------------+---------+
| ``true_sequential``    | bool                             | True    |
+------------------------+----------------------------------+---------+
|``inside_layer_modules``| ``Optional[List[str]]``          | None    |
+------------------------+----------------------------------+---------+
|``model_decoder_layers``| ``Optional[str]``                | None    |
+------------------------+----------------------------------+---------+


Step 4: Setting up the overall ``Config`` for the model.
--------------------------------------------------------

In ``Config``, users should set instances for all information of quantization (all instances are optional except global_quant_config).
For example:

.. code:: python

   # Example 1: W_INT8_A_INT8_PER_TENSOR
   quant_config = Config(global_quant_config=W_INT8_A_INT8_PER_TENSOR_CONFIG)

   # Example 2: W_UINT4_PER_GROUP with advanced algorithm
   quant_config = Config(global_quant_config=W_UINT4_PER_GROUP_CONFIG, algo_config=ALGORITHM_CONFIG)
   EXCLUDE_LAYERS = ["lm_head"] # For language models
   quant_config = replace(quant_config, exclude=EXCLUDE_LAYERS)

   # Example 3: W_FP8_A_FP8_PER_TENSOR with KV_CACHE_FP8
   quant_config = Config(global_quant_config=W_FP8_A_FP8_PER_TENSOR_CONFIG)
   KV_CACHE_CFG = {
       "*v_proj":
       QuantizationConfig(input_tensors=quant_config.global_quant_config.input_tensors,
                          weight=quant_config.global_quant_config.weight,
                          output_tensors=FP8_PER_TENSOR_SPEC),
       "*k_proj":
       QuantizationConfig(input_tensors=quant_config.global_quant_config.input_tensors,
                          weight=quant_config.global_quant_config.weight,
                          output_tensors=FP8_PER_TENSOR_SPEC),
   }
   quant_config = replace(quant_config, layer_quant_config=KV_CACHE_CFG)

For parameter explanation:

.. list-table:: Parameter Explanation
   :widths: 20 40 20 20
   :header-rows: 1

   * - Name
     - Class Type
     - Option
     - Default
   * - ``global_quant_config``
     - ``QuantizationConfig``
     -
     -
   * - ``layer_type_quant_config``
     - ``Dict[Type[nn.Module], QuantizationConfig]``
     -
     - None
