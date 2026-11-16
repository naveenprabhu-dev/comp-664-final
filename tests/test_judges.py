"""Task-head and judge integration checks; tiny actual Qwen3.5, no downloads."""

from dataclasses import replace
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from safetensors.torch import load_file, save_file
from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

from orderbias.models.backbone import prepare_backbone
from orderbias.models.checkpoints import load_judge_checkpoint, save_judge_checkpoint
from orderbias.models.config import ExperimentConfig, JudgeConfig
from orderbias.models.heads import ClassificationHead, ScoringHead, last_non_padding
from orderbias.models.interfaces import TokenBatch
from orderbias.models.judges import PairwiseJudge, SharedScorer, build_judge, load_judge


def tiny_backbone():
    return Qwen3_5TextModel(Qwen3_5TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=8,
        max_position_embeddings=4096,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.5, "mrope_section": [2, 1, 1]},
    ))


def config_for(kind="pairwise", *, lora=True, checkpoint=None):
    backbone = {"compute_dtype": "float32"}
    if checkpoint is not None:
        backbone["checkpoint"] = str(checkpoint)
    return ExperimentConfig.from_dict({
        "backbone": backbone, "judge": {"kind": kind}, "lora": {"enabled": lora},
    })


def batch(rows=None, mask=None):
    ids = torch.tensor(rows if rows is not None else [[1, 2, 3, 4], [5, 6, 7, 8]])
    return TokenBatch(ids, torch.tensor(mask) if mask is not None else torch.ones_like(ids))


def prediction(judge, inputs):
    output = judge(inputs)
    return output.logits if isinstance(judge, PairwiseJudge) else output.scores


def test_pool_selects_last_valid_not_count_minus_one_and_keeps_gradients():
    hidden = torch.arange(24.0).reshape(2, 4, 3).requires_grad_()
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]])
    pooled = last_non_padding(hidden, mask)
    torch.testing.assert_close(pooled, torch.stack([hidden[0, 3], hidden[1, 1]]))
    pooled.sum().backward()
    assert hidden.grad[0, 3].sum() == 3
    assert hidden.grad[1, 1].sum() == 3
    assert hidden.grad.sum() == 6


@pytest.mark.parametrize("mask", [
    [[0, 0, 0, 0]], [[1, 0, 1, 0]], [[1, 2, 1, 0]],
    [[1, float("nan"), 1, 0]],
])
def test_invalid_padding_rejected(mask):
    with pytest.raises(ValueError):
        last_non_padding(torch.ones(1, 4, 3), torch.tensor(mask))


def test_heads_seed_bias_and_fp32_under_autocast():
    first = ClassificationHead(32, JudgeConfig(), seed=7)
    torch.randn(100)  # Global RNG consumption cannot change head initialization
    second = ClassificationHead(32, JudgeConfig(), seed=7)
    torch.testing.assert_close(first.weight, second.weight, rtol=0, atol=0)
    assert torch.count_nonzero(first.bias) == 0
    assert not torch.equal(first.weight, ClassificationHead(32, JudgeConfig(), seed=8).weight)
    scorer = ScoringHead(32, JudgeConfig(kind="shared_scorer", head_bias=False), seed=7)
    assert scorer.bias is None
    pooled = torch.randn(1, 32, dtype=torch.bfloat16, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits, scores = first(pooled), scorer(pooled)
    assert logits.shape == (1, 2) and scores.shape == (1,)
    assert logits.dtype == scores.dtype == torch.float32
    (logits.sum() + scores.sum()).backward()
    assert pooled.grad is not None


@pytest.mark.parametrize("kind", ["pairwise", "shared_scorer"])
@pytest.mark.parametrize("lora", [False, True])
def test_judge_forward_and_optimizer_update_only_heads_and_adapters(kind, lora):
    config = config_for(kind, lora=lora)
    backbone = tiny_backbone()
    original = [(p, p.detach().clone()) for p in backbone.parameters()]
    judge = build_judge(prepare_backbone(backbone, config), config)
    assert isinstance(judge, PairwiseJudge if kind == "pairwise" else SharedScorer)
    inputs = batch()
    pred = prediction(judge, inputs)
    assert pred.shape == ((2, 2) if kind == "pairwise" else (2,))
    head_before = judge.head.weight.detach().clone()
    optimizer = torch.optim.AdamW([p for p in judge.parameters() if p.requires_grad], lr=0.01)
    # Phase 4 supplies the actual objectives; this scalar checks end-to-end flow.
    pred.sum().backward()
    for name, parameter in judge.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        else:
            assert parameter.grad is None, name
    optimizer.step()
    assert not torch.equal(head_before, judge.head.weight)
    assert all(torch.equal(parameter, initial) for parameter, initial in original)
    summary = judge.parameter_summary()
    assert summary["head_parameters"] == (66 if kind == "pairwise" else 33)
    assert summary["trainable_parameters"] == sum(p.numel() for p in judge.parameters() if p.requires_grad)


@pytest.mark.parametrize("kind", ["pairwise", "shared_scorer"])
def test_padding_and_batch_order_do_not_change_predictions(kind):
    torch.manual_seed(9)
    config = config_for(kind)
    judge = build_judge(prepare_backbone(tiny_backbone(), config), config).eval()
    with torch.no_grad():
        for name, parameter in judge.backbone.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.01)  # Exercise nonzero learned corrections
    with torch.no_grad():
        alone = prediction(judge, batch([[1, 2, 3]]))
        # Same short sequence in both padding layouts, with a longer companion.
        left = prediction(judge, batch([[0, 0, 1, 2, 3], [4, 5, 6, 7, 8]],
                                     [[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]]))
        right_batch = batch([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]],
                            [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
        right = prediction(judge, right_batch)
        reversed_pred = prediction(judge, TokenBatch(
            right_batch.input_ids.flip(0), right_batch.attention_mask.flip(0),
        ))
    torch.testing.assert_close(alone[0], left[0], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(alone[0], right[0], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(right.flip(0), reversed_pred, atol=1e-6, rtol=1e-5)
    if kind == "shared_scorer":
        gap = right[0] - right[1]
        swapped_gap = reversed_pred[0] - reversed_pred[1]
        torch.testing.assert_close(gap, -swapped_gap, atol=1e-6, rtol=1e-5)


def test_config_mismatch_and_changed_trainables_are_rejected():
    config = config_for()
    loaded = prepare_backbone(tiny_backbone(), config)
    with pytest.raises(ValueError, match="lora configuration"):
        build_judge(loaded, replace(config, lora=replace(config.lora, rank=16)))
    with pytest.raises(ValueError, match="kind"):
        SharedScorer(loaded, config)
    judge = build_judge(loaded, config)
    judge.head.weight.requires_grad_(False)
    with pytest.raises(ValueError, match="task-head"):
        judge.validate_trainable_parameters()
    judge.head.weight.requires_grad_(True)
    next(p for p in judge.backbone.parameters() if not p.requires_grad).requires_grad_(True)
    with pytest.raises(ValueError, match="Backbone trainables"):
        judge.validate_trainable_parameters()


def test_invalid_token_batch_and_length_rejected():
    config = config_for(lora=False)
    judge = build_judge(prepare_backbone(tiny_backbone(), config), config)
    with pytest.raises(ValueError, match="integer"):
        judge(TokenBatch(torch.ones(1, 3), torch.ones(1, 3)))
    with pytest.raises(ValueError, match="matching shape"):
        judge(TokenBatch(torch.ones(1, 3, dtype=torch.long), torch.ones(1, 2)))
    with pytest.raises(ValueError, match="max_input_length"):
        judge(TokenBatch(torch.ones(1, 4097, dtype=torch.long), torch.ones(1, 4097)))


@pytest.mark.parametrize("kind", ["pairwise", "shared_scorer"])
@pytest.mark.parametrize("lora", [False, True])
def test_trained_judge_checkpoint_roundtrip(kind, lora, tmp_path):
    base_dir = tmp_path / "base"
    tiny_backbone().save_pretrained(base_dir)
    config = config_for(kind, lora=lora, checkpoint=base_dir)
    judge = load_judge(config, local_files_only=True)
    optimizer = torch.optim.AdamW([p for p in judge.parameters() if p.requires_grad], lr=0.01)
    inputs = batch()
    prediction(judge, inputs).sum().backward()
    optimizer.step()
    judge.eval()
    with torch.no_grad():
        before = prediction(judge, inputs)
    directory = tmp_path / "saved"
    save_judge_checkpoint(judge, directory)
    assert not judge.training  # Reporting/saving must not turn dropout back on
    tensors = load_file(str(directory / "trainable.safetensors"))
    assert "head.weight" in tensors and "head.bias" in tensors
    assert all(name.startswith("head.") or "lora_" in name for name in tensors)
    restored = load_judge_checkpoint(directory, local_files_only=True)
    assert not restored.training
    restored.validate_trainable_parameters()
    with torch.no_grad():
        torch.testing.assert_close(before, prediction(restored, inputs), rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        save_judge_checkpoint(judge, directory)
    # Losing the task head must never result in silent random reinitialization.
    del tensors["head.weight"]
    save_file(tensors, str(directory / "trainable.safetensors"))
    with pytest.raises(ValueError, match="exactly"):
        load_judge_checkpoint(directory, local_files_only=True)


def test_checkpoint_pins_resolved_remote_revision(tmp_path):
    config = config_for(lora=False)
    loaded = prepare_backbone(tiny_backbone(), config, resolved_revision="resolved-commit")
    judge = build_judge(loaded, config)
    save_judge_checkpoint(judge, tmp_path / "saved")
    saved_config = json.loads((tmp_path / "saved/resolved_config.json").read_text())
    assert saved_config["backbone"]["revision"] == "resolved-commit"


def test_local_checkpoint_reference_survives_working_directory_change(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tiny_backbone().save_pretrained("base")
    config = config_for(lora=False, checkpoint="base")
    judge = load_judge(config, local_files_only=True)
    save_judge_checkpoint(judge, "saved")
    config_dict = json.loads((tmp_path / "saved/resolved_config.json").read_text())
    assert config_dict["backbone"]["checkpoint"] == str((tmp_path / "base").resolve())
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    restored = load_judge_checkpoint(tmp_path / "saved", local_files_only=True)
    assert isinstance(restored, PairwiseJudge)


def test_bad_head_shape_is_rejected(tmp_path):
    tiny_backbone().save_pretrained(tmp_path / "base")
    config = config_for(lora=False, checkpoint=tmp_path / "base")
    judge = load_judge(config, local_files_only=True)
    save_judge_checkpoint(judge, tmp_path / "saved")
    path = tmp_path / "saved/trainable.safetensors"
    tensors = load_file(str(path))
    tensors["head.weight"] = tensors["head.weight"][:1].clone()
    save_file(tensors, str(path))
    with pytest.raises(ValueError, match="shape/dtype mismatch"):
        load_judge_checkpoint(tmp_path / "saved", local_files_only=True)
