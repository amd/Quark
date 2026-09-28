"""
Simple test for model loading using get_model() function.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from transformers import AutoConfig

from quark.common.utils.import_utils import is_transformers_version_higher_or_equal
from quark.torch.utils.llm import get_model
from quark.torch.utils.llm import model_preparation as model_preparation_module


def test_deepseek_v4_loading_uses_native_config_after_vllm_registration(monkeypatch, tmp_path):
    native = pytest.importorskip("transformers.models.deepseek_v4.configuration_deepseek_v4")
    native.DeepseekV4Config(architectures=["DeepseekV4ForCausalLM"]).save_pretrained(tmp_path)
    shadow_config = native.PreTrainedConfig(model_type="deepseek_v4")
    monkeypatch.setattr(
        model_preparation_module,
        "AutoConfig",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: shadow_config),
    )

    def load_model(path, **kwargs):
        config = kwargs.get("config", shadow_config)
        assert isinstance(config, native.DeepseekV4Config)
        assert config.pad_token_id is None
        assert len(config.layer_types) == config.num_hidden_layers
        model = torch.nn.Linear(1, 1, device="meta")
        model.config = config
        return model

    loader = MagicMock(side_effect=load_model)
    monkeypatch.setattr(model_preparation_module, "AutoModelForCausalLM", SimpleNamespace(from_pretrained=loader))
    model, _ = get_model(str(tmp_path), device="meta")
    assert isinstance(model.config, native.DeepseekV4Config)
    loader.assert_called_once()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not is_transformers_version_higher_or_equal("5.2.0"),
    reason="Qwen3.5 model_type=qwen3_5 requires transformers >= 5.2.0",
)
def test_get_model_qwen35_0_8b():
    """Test loading Qwen/Qwen3.5-0.8B model using get_model()."""
    model_id = "Qwen/Qwen3.5-0.8B"

    config = AutoConfig.from_pretrained(model_id)

    # Load model with default parameters
    model, model_dtype = get_model(
        ckpt_path=model_id,
        data_type="auto",
        device="cuda",
        multi_gpu=False,
        multi_device=False,
        attn_implementation="eager",
        trust_remote_code=True,
    )

    # Verify model is loaded correctly
    assert model is not None, "Model should not be None"
    assert isinstance(model, torch.nn.Module), "Model should be a torch.nn.Module"

    # Verify model is in eval mode
    assert not model.training, "Model should be in eval mode"

    # Verify model dtype
    assert model_dtype in [torch.float16, torch.bfloat16, torch.float32], f"Unexpected dtype: {model_dtype}"

    # Verify model config
    assert hasattr(model, "config"), "Model should have a config attribute"
    assert model.config._name_or_path == model_id, "Model config should have correct _name_or_path"

    # Verify model has parameters
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 0, "Model should have parameters"

    assert model.config.model_type == config.model_type


@pytest.mark.parametrize(
    ("ckpt_path", "model_type"),
    [
        pytest.param("MiniMaxAI/MiniMax-M3-VL", "minimax_m3_vl", id="minimax_m3_vl"),
        pytest.param("meta-models/Muse-Glimmer-30B", "muse_glimmer", id="muse_glimmer"),
    ],
)
def test_get_model_vlm_uses_image_text_loader(monkeypatch, ckpt_path, model_type):
    """VLM wrappers (MiniMax-M3-VL, Muse-Glimmer) load via AutoModelForImageTextToText
    rather than AutoModelForCausalLM."""
    config = SimpleNamespace(model_type=model_type)
    calls = {}

    class DummyAutoConfig:
        """Dummy AutoConfig class for testing purposes."""

        @staticmethod
        def from_pretrained(path, **kwargs):
            """Mock from_pretrained method that records calls.
            Args:
                path: Path to the model checkpoint.
                **kwargs: Additional keyword arguments.
            Returns:
                SimpleNamespace: A config object with model_type attribute.
            """
            calls["config"] = {"path": path, **kwargs}
            return config

    class DummyModel(torch.nn.Module):
        """Dummy model for testing purposes.
        This is a minimal torch.nn.Module implementation used for testing
        model loading functionality without requiring actual model weights.
        """

        def __init__(self):
            """Initialize the DummyModel.
            Creates a minimal model with a simple config, a single weight parameter,
            and initializes registered_auto_class to None.
            """
            super().__init__()
            self.config = SimpleNamespace(_name_or_path="")
            self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
            self.registered_auto_class = None

        def register_for_auto_class(self, auto_class):
            """Register the model for auto class.
            Args:
                auto_class: The auto class to register the model with.
            """
            self.registered_auto_class = auto_class

    class DummyImageTextLoader:
        """Dummy class to mock AutoModelForImageTextToText loader for testing."""

        @staticmethod
        def from_pretrained(path, **kwargs):
            """
            Mock the from_pretrained method to capture call arguments.
            Args:
                path: The model checkpoint path.
                **kwargs: Additional keyword arguments passed to the loader.
            Returns:
                DummyModel: A dummy model instance for testing.
            """
            calls["image_text_loader"] = {"path": path, **kwargs}
            return DummyModel()

    monkeypatch.setattr(model_preparation_module, "AutoConfig", DummyAutoConfig)
    monkeypatch.setattr(model_preparation_module, "AutoModelForImageTextToText", DummyImageTextLoader)
    monkeypatch.setattr(model_preparation_module, "_is_compressed_tensors_model", lambda _: False)
    monkeypatch.setattr(model_preparation_module, "_set_hf_loader_workers", lambda _: "saved-workers")
    monkeypatch.setattr(
        model_preparation_module,
        "_restore_hf_loader_workers",
        lambda saved: calls.setdefault("restored", saved),
    )

    model, model_dtype = get_model(
        ckpt_path=ckpt_path,
        data_type="float32",
        device="cpu",
        multi_gpu=False,
        multi_device=False,
        attn_implementation="eager",
        trust_remote_code=True,
    )

    assert calls["config"] == {
        "path": ckpt_path,
        "trust_remote_code": True,
        "attn_implementation": "eager",
    }
    assert calls["image_text_loader"] == {
        "path": ckpt_path,
        "device_map": "cpu",
        "torch_dtype": torch.float32,
        "max_memory": None,
        "trust_remote_code": True,
        "attn_implementation": "eager",
    }
    assert calls["restored"] == "saved-workers"
    assert model.config._name_or_path == ckpt_path
    assert model.registered_auto_class is None
    assert not model.training
    assert model_dtype is torch.float32


@pytest.mark.parametrize(
    ("multi_gpu", "multi_device", "expected_device_map", "expected_max_memory_calls"),
    [
        ("auto", False, "auto", 1),
        (False, False, "cpu", 0),
        # multi_device already populated max_memory; multi_gpu must not recompute over it.
        ("auto", True, "auto", 1),
    ],
)
def test_get_model_multi_gpu_auto_populates_max_memory(
    monkeypatch, multi_gpu, multi_device, expected_device_map, expected_max_memory_calls
):
    """``--multi_gpu auto`` must populate max_memory via get_device_max_memory() when the
    caller didn't already supply one, so device_map="auto" inference is forced to actually
    split the model instead of greedily cramming it onto one device."""
    config = SimpleNamespace(model_type="opt")
    sentinel_max_memory = {0: "1.0GB", 1: "2.0GB", "cpu": "10.0GB"}
    calls = {"max_memory": 0}

    def fake_get_device_max_memory():
        calls["max_memory"] += 1
        return sentinel_max_memory

    class DummyAutoConfig:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return config

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(_name_or_path="")
            self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))

    class DummyCausalLMLoader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["causal_lm_loader"] = {"path": path, **kwargs}
            return DummyModel()

    monkeypatch.setattr(model_preparation_module, "AutoConfig", DummyAutoConfig)
    monkeypatch.setattr(model_preparation_module, "AutoModelForCausalLM", DummyCausalLMLoader)
    monkeypatch.setattr(model_preparation_module, "get_device_max_memory", fake_get_device_max_memory)
    monkeypatch.setattr(model_preparation_module, "_is_compressed_tensors_model", lambda _: False)
    monkeypatch.setattr(model_preparation_module, "_set_hf_loader_workers", lambda _: "saved-workers")
    monkeypatch.setattr(model_preparation_module, "_restore_hf_loader_workers", lambda saved: None)

    get_model(
        ckpt_path="dummy-ckpt",
        data_type="float32",
        device="cpu",
        multi_gpu=multi_gpu,
        multi_device=multi_device,
        attn_implementation="eager",
        trust_remote_code=True,
    )

    expected_max_memory = sentinel_max_memory if expected_max_memory_calls else None
    assert calls["causal_lm_loader"]["max_memory"] == expected_max_memory
    assert calls["causal_lm_loader"]["device_map"] == expected_device_map
    assert calls["max_memory"] == expected_max_memory_calls


def test_get_model_compressed_tensors_restores_fp32_params_when_auto(monkeypatch):
    """
    The compressed-tensors loading branch (used by e.g. Kimi-K2.5) returns early, before the
    generic `AutoModelForCausalLM.from_pretrained` path's `_restore_fp32_params_from_source` call.
    `get_model` must also invoke it in this branch when `data_type="auto"`, otherwise fp32
    parameters downcast during compressed-tensors loading (e.g. `e_score_correction_bias`) are
    never restored.
    """
    ckpt_path = "moonshotai/Kimi-K2.5"
    config = SimpleNamespace(model_type="kimi_k25")
    calls = {}

    class DummyAutoConfig:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return config

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(_name_or_path="")
            self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))

    def dummy_load_from_compressed_tensors(**kwargs):
        calls["load_from_compressed_tensors"] = kwargs
        return DummyModel()

    monkeypatch.setattr(model_preparation_module, "AutoConfig", DummyAutoConfig)
    monkeypatch.setattr(model_preparation_module, "_is_compressed_tensors_model", lambda _: True)
    monkeypatch.setattr(model_preparation_module, "_load_from_compressed_tensors", dummy_load_from_compressed_tensors)
    monkeypatch.setattr(
        model_preparation_module,
        "_restore_fp32_params_from_source",
        lambda model, path: calls.setdefault("restore_fp32_params", (model, path)),
    )

    model, model_dtype = get_model(
        ckpt_path=ckpt_path,
        data_type="auto",
        device="cpu",
        multi_gpu=False,
        multi_device=False,
        attn_implementation="eager",
        trust_remote_code=True,
    )

    assert "restore_fp32_params" in calls, (
        "_restore_fp32_params_from_source must be called for the compressed-tensors branch"
    )
    assert calls["restore_fp32_params"] == (model, ckpt_path)
    assert model_dtype is None


def test_get_model_compressed_tensors_skips_restore_fp32_params_when_not_auto(monkeypatch):
    """When data_type is not "auto", the compressed-tensors branch must not attempt to restore fp32 params."""
    ckpt_path = "moonshotai/Kimi-K2.5"
    config = SimpleNamespace(model_type="kimi_k25")
    calls = {}

    class DummyAutoConfig:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return config

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(_name_or_path="")
            self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))

    monkeypatch.setattr(model_preparation_module, "AutoConfig", DummyAutoConfig)
    monkeypatch.setattr(model_preparation_module, "_is_compressed_tensors_model", lambda _: True)
    monkeypatch.setattr(model_preparation_module, "_load_from_compressed_tensors", lambda **kwargs: DummyModel())
    monkeypatch.setattr(
        model_preparation_module,
        "_restore_fp32_params_from_source",
        lambda model, path: calls.setdefault("restore_fp32_params", (model, path)),
    )

    get_model(
        ckpt_path=ckpt_path,
        data_type="bfloat16",
        device="cpu",
        multi_gpu=False,
        multi_device=False,
        attn_implementation="eager",
        trust_remote_code=True,
    )

    assert "restore_fp32_params" not in calls


def test_get_model_compressed_tensors_skips_restore_fp32_params_on_meta(monkeypatch):
    """A meta-only structural load must not reopen checkpoint shards to restore fp32 values."""
    ckpt_path = "moonshotai/Kimi-K2.5"
    config = SimpleNamespace(model_type="kimi_k25")
    calls = {}

    class DummyAutoConfig:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return config

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(_name_or_path="")
            self.weight = torch.nn.Parameter(torch.empty(1, device="meta"))

    def dummy_load_from_compressed_tensors(**kwargs):
        calls["load_from_compressed_tensors"] = kwargs
        return DummyModel()

    monkeypatch.setattr(model_preparation_module, "AutoConfig", DummyAutoConfig)
    monkeypatch.setattr(model_preparation_module, "_is_compressed_tensors_model", lambda _: True)
    monkeypatch.setattr(model_preparation_module, "_load_from_compressed_tensors", dummy_load_from_compressed_tensors)
    monkeypatch.setattr(
        model_preparation_module,
        "_restore_fp32_params_from_source",
        lambda model, path: calls.setdefault("restore_fp32_params", (model, path)),
    )

    model, model_dtype = get_model(
        ckpt_path=ckpt_path,
        data_type="auto",
        device="meta",
        multi_gpu=False,
        multi_device=False,
        attn_implementation="eager",
        trust_remote_code=True,
    )

    assert calls["load_from_compressed_tensors"]["device_map"] == "meta"
    assert "restore_fp32_params" not in calls
    assert model.weight.device.type == "meta"
    assert model_dtype is None


class _FakeShardHandle:
    """Stand-in for a ``safe_open`` handle exposing one weight key."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def keys(self):
        return ["weight"]

    def get_tensor(self, key):
        return torch.zeros(1)


def _patch_ct_skeleton(monkeypatch, loading_module, dummy_model, config, calls):
    """Stub the skeleton-building dependencies of ``_load_from_compressed_tensors``.

    Everything up to (and including) ``compressor.compress_model`` is replaced so the
    real loader body runs without any model download, while the checkpoint-I/O and
    dispatch primitives are spied on so tests can assert whether they were reached.
    """
    import contextlib

    class DummyAutoModel:
        @staticmethod
        def from_config(cfg, **kwargs):
            return dummy_model

    class DummyCompressor:
        quantization_config = SimpleNamespace()

        def compress_model(self, model):
            pass

    class DummyModelCompressor:
        @staticmethod
        def from_compression_config(compression_config):
            return DummyCompressor()

    class DummyCompressedTensorsConfig:
        @staticmethod
        def from_dict(quantization_config):
            return SimpleNamespace()

    def fake_safe_open(path, framework="pt", device="cpu"):
        calls["safe_open"] = calls.get("safe_open", 0) + 1
        return _FakeShardHandle()

    def fake_set_module_tensor_to_device(model, key, device, value=None):
        calls["set_tensor"] = calls.get("set_tensor", 0) + 1

    def fake_dispatch_model(model, device_map):
        calls["dispatch"] = calls.get("dispatch", 0) + 1

    # Some names are only bound when compressed-tensors / accelerate are installed;
    # raising=False lets the stubs stand in even when those packages are absent.
    monkeypatch.setattr(loading_module, "is_compressed_tensors_available", lambda: True)
    monkeypatch.setattr(loading_module, "is_accelerate_available", lambda: True)
    monkeypatch.setattr(loading_module, "is_package_lower_or_equal", lambda *a, **k: False)
    monkeypatch.setattr(loading_module, "no_init_weights", contextlib.nullcontext, raising=False)
    monkeypatch.setattr(loading_module, "init_empty_weights", contextlib.nullcontext, raising=False)
    monkeypatch.setattr(loading_module, "AutoModelForCausalLM", DummyAutoModel, raising=False)
    monkeypatch.setattr(loading_module, "ModelCompressor", DummyModelCompressor, raising=False)
    monkeypatch.setattr(loading_module, "CompressedTensorsConfig", DummyCompressedTensorsConfig, raising=False)
    monkeypatch.setattr(loading_module, "apply_quantization_config", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(loading_module, "safe_open", fake_safe_open)
    monkeypatch.setattr(loading_module, "set_module_tensor_to_device", fake_set_module_tensor_to_device, raising=False)
    monkeypatch.setattr(loading_module, "dispatch_model", fake_dispatch_model, raising=False)
    monkeypatch.setattr(loading_module.GlobalProfiler, "log_torch_memory", lambda *a, **k: None)


def test_load_from_compressed_tensors_meta_skips_io(monkeypatch, tmp_path):
    """A meta device_map must return the structural model without reading shards or dispatching."""
    from quark.torch.integrations.compressed_tensors import loading as ct_loading_module

    config = SimpleNamespace(model_type="llama", quantization_config={})

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(1, device="meta"))

    dummy_model = DummyModel()
    calls: dict = {}
    _patch_ct_skeleton(monkeypatch, ct_loading_module, dummy_model, config, calls)

    model = ct_loading_module._load_from_compressed_tensors(
        model_dir=str(tmp_path),
        config=config,
        device_map="meta",
        max_memory=None,
        trust_remote_code=True,
    )

    assert model is dummy_model
    assert model.weight.device.type == "meta"
    assert calls.get("safe_open", 0) == 0
    assert calls.get("set_tensor", 0) == 0
    assert calls.get("dispatch", 0) == 0


def test_load_from_compressed_tensors_normal_device_reads_shards(monkeypatch, tmp_path):
    """A non-meta device_map must follow the existing load path: read shards and move weights."""
    from quark.torch.integrations.compressed_tensors import loading as ct_loading_module

    config = SimpleNamespace(model_type="llama", quantization_config={})

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Already on cpu so the post-load meta guard passes with the stubbed setter.
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.to_calls: list = []

        def to(self, *args, **kwargs):
            self.to_calls.append((args, kwargs))
            return self

    dummy_model = DummyModel()
    calls: dict = {}
    _patch_ct_skeleton(monkeypatch, ct_loading_module, dummy_model, config, calls)

    # A real (empty) shard file so the loader's stat() succeeds; contents are unused
    # because safe_open is stubbed.
    (tmp_path / "model.safetensors").write_bytes(b"")

    model = ct_loading_module._load_from_compressed_tensors(
        model_dir=str(tmp_path),
        config=config,
        device_map="cpu",
        max_memory=None,
        trust_remote_code=True,
    )

    assert model is dummy_model
    assert calls.get("safe_open", 0) >= 1
    assert calls.get("set_tensor", 0) >= 1
    assert calls.get("dispatch", 0) == 0
    assert model.to_calls == [(("cpu",), {})]


@pytest.mark.parametrize(
    ("device_map", "expected"),
    [
        (torch.device("meta"), True),
        (torch.device("cpu"), False),
        ("meta", True),
        ("meta:0", True),
        ("cpu", False),
        # An unparsable string falls into the ``except RuntimeError`` branch.
        ("not-a-real-device", False),
        # Non-str / non-device inputs fall through to the final equality check.
        ({"": "cpu"}, False),
        (None, False),
    ],
)
def test_is_meta_device_map(device_map, expected):
    """Cover every branch of ``_is_meta_device_map`` (torch.device, str, and fallthrough)."""
    from quark.torch.integrations.compressed_tensors.loading import _is_meta_device_map

    assert _is_meta_device_map(device_map) is expected


@pytest.mark.parametrize(
    ("num_calib_data", "seq_len", "expected"),
    [
        # Budget allows 393216 tokens: 393216 // 2048 = 192, so 128 samples is the binding limit.
        (128, 2048, 128),
        # More samples than the budget fits: the budget wins.
        (1000, 2048, 192),
        # A sequence longer than the whole budget still has to run one sample at a time.
        (128, 1_000_000, 1),
    ],
)
def test_get_per_block_calib_batch_size(num_calib_data, seq_len, expected):
    """The batch is the smaller of the sample count and the activation budget, never below 1."""
    from quark.torch.utils.llm import get_per_block_calib_batch_size

    assert get_per_block_calib_batch_size(num_calib_data, seq_len) == expected


class _DecoderBlock(torch.nn.Module):
    """One decoder block whose weight is *numel* float32 elements (``numel * 4`` bytes)."""

    def __init__(self, numel: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(numel))


class _DecoderStackCalibModel(torch.nn.Module):
    """Model exposing a ``layers`` ``nn.ModuleList`` -- what ``infer_decoder_layers_path`` looks for.

    Defaults to a single 400-byte (100 float32 elements) block and no non-block weights.
    """

    def __init__(self, block_numels: tuple[int, ...] = (100,), embed_numel: int = 0):
        super().__init__()
        self.layers = torch.nn.ModuleList([_DecoderBlock(n) for n in block_numels])
        if embed_numel:
            # Stands in for embed_tokens / lm_head / final norm: outside the decoder stack, but
            # pinned to the target device by per_block_runner.prepare() for the whole run.
            self.embed_tokens = torch.nn.Parameter(torch.zeros(embed_numel))


def test_per_block_resident_bytes_uses_largest_block_not_last():
    """Interleaved dense/MoE stacks can end on a dense block, so size off the largest one."""
    from quark.torch.utils.llm.model_preparation import _per_block_resident_bytes

    # Middle block: 100 elements (400 bytes, "moe"). Last block: 10 elements (40 bytes, "dense").
    # Measuring the last block would give 40; averaging would give (40+400+40)/3=160.
    model = _DecoderStackCalibModel(block_numels=(10, 100, 10))
    assert _per_block_resident_bytes(model) == (400, 0)


def test_per_block_resident_bytes_reports_non_block_weights():
    """Weights outside the decoder stack stay on the device for the whole run, so report them."""
    from quark.torch.utils.llm.model_preparation import _per_block_resident_bytes

    model = _DecoderStackCalibModel(block_numels=(100,), embed_numel=50)
    assert _per_block_resident_bytes(model) == (400, 200)


def test_per_block_resident_bytes_returns_none_without_decoder_stack():
    """No locatable decoder-block nn.ModuleList -> None, so callers fall back to the static budget."""
    from quark.torch.utils.llm.model_preparation import _per_block_resident_bytes

    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.zeros(10))
    assert _per_block_resident_bytes(model) is None


def test_get_per_block_calib_batch_size_falls_back_on_non_cuda_device():
    """A non-CUDA device has no free-memory query to make, so the static budget applies."""
    from quark.torch.utils.llm import get_per_block_calib_batch_size

    model = _DecoderStackCalibModel()
    assert get_per_block_calib_batch_size(128, 2048, device="cpu", model=model) == 128


def test_get_per_block_calib_batch_size_falls_back_without_model():
    """Without a model there's nothing to size a decoder block against, so use the static budget."""
    from quark.torch.utils.llm import get_per_block_calib_batch_size

    assert get_per_block_calib_batch_size(128, 2048, device=torch.device("cuda:0")) == 128


def test_get_per_block_calib_batch_size_uses_device_free_memory(monkeypatch):
    """The budget is derived from the device's actual free memory, not the static estimate."""
    from quark.torch.utils.llm import get_per_block_calib_batch_size

    model = _DecoderStackCalibModel()
    # One decoder block is 400 bytes; reserve = free - 2*400 (streamed block headroom).
    # free=2_048_800 -> reserve=2_048_000 -> 2_048_000 // (400 KiB) == 5 tokens -> batch 5 at seq_len=1.
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (2_048_800, 1 << 40))

    assert get_per_block_calib_batch_size(100, 1, device=torch.device("cuda:0"), model=model) == 5


def test_get_per_block_calib_batch_size_subtracts_gpu_resident_blocks(monkeypatch):
    """Decoder blocks kept GPU-resident shrink the budget left for calibration activations."""
    from quark.torch.utils.llm import get_per_block_calib_batch_size

    model = _DecoderStackCalibModel()
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (2_048_800, 1 << 40))

    # Same free memory as above, but 2 resident blocks (2*400 bytes) eat into the budget:
    # reserve = 2_048_800 - 2*400 (resident) - 2*400 (streamed) == 2_047_200 -> 4 tokens, not 5.
    batch = get_per_block_calib_batch_size(100, 1, device=torch.device("cuda:0"), model=model, n_gpu_resident_blocks=2)
    assert batch == 4


def test_get_per_block_calib_batch_size_subtracts_non_block_weights(monkeypatch):
    """prepare() pins embeddings/lm_head to the device too, so they come out of the budget."""
    from quark.torch.utils.llm import get_per_block_calib_batch_size

    # 400-byte block as above, plus 800 bytes (200 float32 elements) of non-block weights.
    model = _DecoderStackCalibModel(block_numels=(100,), embed_numel=200)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (2_048_800, 1 << 40))

    # reserve = 2_048_800 - 800 (non-block) - 2*400 (streamed) == 2_047_200 -> 4 tokens, not 5.
    assert get_per_block_calib_batch_size(100, 1, device=torch.device("cuda:0"), model=model) == 4


class _RecordingModel(torch.nn.Module):
    """Model that records ``.to`` targets instead of performing a real device move."""

    def __init__(self, numel: int = 4):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(numel))
        self.register_buffer("bias", torch.zeros(numel))
        self.to_calls: list = []

    def to(self, *args, **kwargs):
        self.to_calls.append(args)
        return self


def test_move_model_to_device_if_it_fits_non_cuda_always_moves():
    """A non-cuda target has no free-memory query to make, so the move is unconditional."""
    from quark.torch.utils.llm import move_model_to_device_if_it_fits

    model = _RecordingModel()
    device = torch.device("cpu")

    assert move_model_to_device_if_it_fits(model, device) is model
    assert model.to_calls == [(device,)]


def test_move_model_to_device_if_it_fits_moves_when_weights_fit(monkeypatch):
    """Weights well under the free memory are moved to the accelerator."""
    from quark.torch.utils.llm import move_model_to_device_if_it_fits

    model = _RecordingModel()
    device = torch.device("cuda:0")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (1 << 30, 1 << 30))

    assert move_model_to_device_if_it_fits(model, device) is model
    assert model.to_calls == [(device,)]


def test_move_model_to_device_if_it_fits_keeps_model_when_weights_do_not_fit(monkeypatch):
    """Weights within 10% of the free memory leave the model where it is."""
    from quark.torch.utils.llm import move_model_to_device_if_it_fits

    model = _RecordingModel()
    device = torch.device("cuda:0")
    # 8 fp32 params/buffers = 32 bytes; 33 bytes free leaves no 10% headroom.
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (33, 1 << 30))

    assert move_model_to_device_if_it_fits(model, device) is model
    assert model.to_calls == []


class _DeviceTrackingModel(_RecordingModel):
    """Recording model that also exposes ``.device``, like a real HF model does."""

    def __init__(self, initial_device: str = "cpu"):
        super().__init__()
        self.device = torch.device(initial_device)


def _explode(_device):
    raise AssertionError("free memory should not be queried for an already-resident model")


def test_move_model_to_device_if_it_fits_no_op_when_already_resident(monkeypatch):
    """A model already on the target device is left alone, without a free-memory query."""
    from quark.torch.utils.llm import move_model_to_device_if_it_fits

    model = _DeviceTrackingModel(initial_device="cuda:0")
    monkeypatch.setattr(torch.cuda, "mem_get_info", _explode)

    assert move_model_to_device_if_it_fits(model, torch.device("cuda:0")) is model
    assert model.to_calls == []


def test_move_model_to_device_if_it_fits_no_op_for_indexless_target(monkeypatch):
    """``cuda`` as a target matches a model on ``cuda:0`` -- that is what ``--device`` looks like."""
    from quark.torch.utils.llm import move_model_to_device_if_it_fits

    model = _DeviceTrackingModel(initial_device="cuda:0")
    monkeypatch.setattr(torch.cuda, "mem_get_info", _explode)

    assert move_model_to_device_if_it_fits(model, torch.device("cuda")) is model
    assert model.to_calls == []


def test_move_model_to_device_if_it_fits_moves_across_device_indices(monkeypatch):
    """A different index on the same device type is still a move, subject to the fit check."""
    from quark.torch.utils.llm import move_model_to_device_if_it_fits

    model = _DeviceTrackingModel(initial_device="cuda:0")
    device = torch.device("cuda:1")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (1 << 30, 1 << 30))

    assert move_model_to_device_if_it_fits(model, device) is model
    assert model.to_calls == [(device,)]
