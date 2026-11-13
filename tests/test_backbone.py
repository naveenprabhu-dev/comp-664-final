"""Offline integration checks using tiny, real hybrid Qwen3.5 text models.

Requires requirements-training.txt; the lightweight suite skips this module.
No pretrained weights, tokenizer, GPU, or network are needed.
"""

import json
from pathlib import Path
import runpy
import sys

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from safetensors.torch import save_file
from transformers import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5TextModel

from orderbias.models.backbone import load_backbone, prepare_backbone, resolve_lora_targets
from orderbias.models.config import ExperimentConfig, LoRAConfig


def tiny_config():
    return Qwen3_5TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=8,
        max_position_embeddings=4096,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.5, "mrope_section": [2, 1, 1]},
    )


def experiment(**sections):
    return ExperimentConfig.from_dict({"backbone": {"compute_dtype": "float32"}, **sections})


def test_presets_cover_both_attention_families_and_mlp():
    model = Qwen3_5TextModel(tiny_config())
    attention = resolve_lora_targets(model, LoRAConfig(target_preset="attention"))
    all_linear = resolve_lora_targets(model, LoRAConfig())
    assert len(attention) == 9  # 5 DeltaNet projections + 4 full attention
    assert len(all_linear) == 15  # Plus 3 MLP projections per layer
    assert "layers.0.linear_attn.in_proj_a" in attention
    assert "layers.1.self_attn.q_proj" in attention
    assert set(attention) < set(all_linear)
    assert not any("conv1d" in name or "embed" in name for name in all_linear)
    config = LoRAConfig(exclude_modules=("layers.1.self_attn.q_proj",))
    assert len(resolve_lora_targets(model, config)) == 14


@pytest.mark.parametrize("target", ["q_proj", "lm_head", "embed_tokens", "layers.0.linear_attn.conv1d"])
def test_invalid_targets_fail_before_attachment(target):
    model = Qwen3_5TextModel(tiny_config())
    with pytest.raises(ValueError, match="ineligible"):
        resolve_lora_targets(model, LoRAConfig(target_preset=None, target_modules=(target,)))
    with pytest.raises(ValueError, match="ineligible"):
        resolve_lora_targets(model, LoRAConfig(exclude_modules=(target,)))


def test_empty_or_non_text_selection_fails():
    model = Qwen3_5TextModel(tiny_config())
    targets = resolve_lora_targets(model, LoRAConfig())
    with pytest.raises(ValueError, match="no language"):
        resolve_lora_targets(model, LoRAConfig(exclude_modules=targets))
    with pytest.raises(ValueError, match="bare"):
        resolve_lora_targets(torch.nn.Linear(2, 2), LoRAConfig())


@pytest.mark.parametrize("checkpointing", [False, True])
def test_real_hybrid_forward_backward_freezes_original_weights(checkpointing, tmp_path):
    torch.manual_seed(7)
    model = Qwen3_5TextModel(tiny_config())
    original = [(parameter, parameter.detach().clone()) for parameter in model.parameters()]
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    mask = torch.ones_like(ids)
    model.eval()
    with torch.no_grad():
        before = model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    config = experiment(backbone={"compute_dtype": "float32", "gradient_checkpointing": checkpointing})
    loaded = prepare_backbone(model, config)
    loaded.model.eval()
    with torch.no_grad():
        after = loaded.model(input_ids=ids, attention_mask=mask).last_hidden_state
    torch.testing.assert_close(before, after)  # Zero initial LoRA correction
    loaded.model.train()
    trainable = [p for p in loaded.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=0.01)
    for _ in range(2):
        optimizer.zero_grad()
        hidden = loaded.model(input_ids=ids, attention_mask=mask).last_hidden_state
        assert hidden.shape == (2, 4, 32)
        hidden[:, -1, 0].sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
        assert any(p.grad.abs().sum() > 0 for p in trainable)
        assert all(p.grad is None for p, _ in original)
        optimizer.step()
    assert all(torch.equal(p, initial) for p, initial in original)
    assert all(p.dtype == torch.float32 for p in trainable)
    assert len(loaded.report.adapters) == 15
    assert loaded.report.trainable_parameters == sum(p.numel() for p in trainable)
    loaded.report.save_json(tmp_path / "report.json")
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["adapters"][0]["a_shape"][0] == config.lora.rank
    assert "transformers" in report["versions"]
    with pytest.raises(ValueError, match="bare|already"):
        prepare_backbone(loaded.model, config)


def test_explicit_targets_and_head_only():
    model = Qwen3_5TextModel(tiny_config())
    loaded = prepare_backbone(model, experiment(lora={
        "target_preset": None, "target_modules": ["layers.1.self_attn.q_proj"],
    }))
    assert len(loaded.report.adapters) == 1
    assert len(loaded.report.trainable_parameter_names) == 2
    probe = prepare_backbone(Qwen3_5TextModel(tiny_config()), experiment(lora={"enabled": False}))
    assert probe.report.trainable_parameters == 0
    assert not any(p.requires_grad for p in probe.model.parameters())
    # The task head will be introduced in Phase 3.


def write_full_checkpoint(path, *, missing=False):
    torch.manual_seed(11)
    model = Qwen3_5TextModel(tiny_config())
    weights = {f"model.language_model.{key}": value.contiguous() for key, value in model.state_dict().items()}
    weights["model.visual.unused.weight"] = torch.ones(2, 2)
    weights["mtp.unused.weight"] = torch.ones(2, 2)
    if missing:
        del weights["model.language_model.layers.0.mlp.up_proj.weight"]
    Qwen3_5Config(text_config=tiny_config().to_dict()).save_pretrained(path)
    save_file(weights, str(path / "model.safetensors"), metadata={"format": "pt"})
    return model


def test_load_multimodal_checkpoint_text_weights_only(tmp_path):
    original = write_full_checkpoint(tmp_path)
    config = experiment(backbone={"checkpoint": str(tmp_path), "compute_dtype": "float32"},
                        lora={"enabled": False})
    loaded = load_backbone(config, local_files_only=True)
    assert loaded.report.hidden_size == 32
    assert not hasattr(loaded.model, "visual")
    assert not hasattr(loaded.model, "lm_head")
    assert "model.visual.unused.weight" in loaded.report.ignored_checkpoint_keys
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, loaded.model.state_dict()[key], rtol=0, atol=0)


def test_missing_language_weight_is_rejected(tmp_path):
    write_full_checkpoint(tmp_path, missing=True)
    config = experiment(backbone={"checkpoint": str(tmp_path), "compute_dtype": "float32"})
    with pytest.raises(RuntimeError, match="Incomplete"):
        load_backbone(config, local_files_only=True)


def test_load_standalone_text_checkpoint_with_lora(tmp_path):
    Qwen3_5TextModel(tiny_config()).save_pretrained(tmp_path)
    config = experiment(backbone={"checkpoint": str(tmp_path), "compute_dtype": "float32"})
    loaded = load_backbone(config, local_files_only=True)
    assert loaded.report.trainable_parameters > 0
    output = loaded.model(input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.ones(1, 3))
    assert torch.isfinite(output.last_hidden_state).all()


def test_smoke_script_writes_results_for_local_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    Qwen3_5TextModel(tiny_config()).save_pretrained(checkpoint)
    model_config = tmp_path / "model.json"
    model_config.write_text(json.dumps({"backbone": {"checkpoint": str(checkpoint)}}))
    root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", [
        "check_backbone.py", "--model-config", str(model_config),
        "--experiment-config", str(root / "configs/experiments/pairwise.yaml"),
        "--local-files-only", "--output-dir", str(output_dir),
    ])
    runpy.run_path(str(root / "scripts/check_backbone.py"), run_name="__main__")
    summary = json.loads((output_dir / "smoke_result.json").read_text())
    assert summary["status"] == "passed"
    assert summary["output_shape"] == [1, 8, 32]
    assert summary["adapter_modules"] == 15
    assert (output_dir / "backbone_report.json").is_file()
    assert json.loads((output_dir / "resolved_config.json").read_text())["backbone"]["compute_dtype"] == "float32"


def test_incompatible_checkpoint_type_and_position_limit(tmp_path):
    transformers.BertConfig().save_pretrained(tmp_path)
    config = experiment(backbone={"checkpoint": str(tmp_path), "compute_dtype": "float32"})
    with pytest.raises(ValueError, match="Unsupported checkpoint"):
        load_backbone(config, local_files_only=True)
    text_config = tiny_config()
    text_config.max_position_embeddings = 2048
    text_config.save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="positional limit"):
        load_backbone(config, local_files_only=True)


def test_4bit_cpu_rejected_before_download():
    config = experiment(quantization={"mode": "4bit", "compute_dtype": "float32"})
    with pytest.raises(ValueError, match="CUDA only"):
        load_backbone(config, device="cpu", local_files_only=True)
    with pytest.raises(ValueError, match="quantization"):
        prepare_backbone(Qwen3_5TextModel(tiny_config()), config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for actual 4-bit execution")
def test_cuda_4bit_load_and_adapter_gradients(tmp_path):
    pytest.importorskip("bitsandbytes")
    Qwen3_5TextModel(tiny_config()).save_pretrained(tmp_path)
    config = experiment(
        backbone={"checkpoint": str(tmp_path), "compute_dtype": "float16", "gradient_checkpointing": True},
        quantization={"mode": "4bit", "compute_dtype": "float16"},
    )
    loaded = load_backbone(config, device="cuda:0", local_files_only=True)
    base = loaded.model.get_base_model()
    assert base.is_loaded_in_4bit
    assert any(module.__class__.__name__ == "Linear4bit" for module in base.modules())
    ids = torch.tensor([[1, 2, 3, 4]], device="cuda:0")
    output = loaded.model(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state
    assert torch.isfinite(output).all()
    output[:, -1, 0].sum().backward()
    for name, parameter in loaded.model.named_parameters():
        if parameter.requires_grad:
            assert "lora_" in name
            assert parameter.dtype == torch.float32
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        else:
            assert parameter.grad is None
