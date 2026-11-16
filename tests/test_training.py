"""Offline tokenization, preference objectives, and actual tiny training runs."""

from dataclasses import replace
from itertools import islice
import json
from pathlib import Path
import runpy
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoTokenizer, PreTrainedTokenizerFast, Qwen3_5TextConfig, Qwen3_5TextModel

from orderbias.data.schema import UnifiedExample
from orderbias.models.backbone import prepare_backbone
from orderbias.models.checkpoints import load_judge_checkpoint
from orderbias.models.config import ExperimentConfig
from orderbias.models.judges import build_judge
from orderbias.models.judges import load_judge
from orderbias.training.data import (
    JudgeEncoder, collate_views, load_tokenizer, presentation_stream,
    read_examples, truncate_text, validate_training_splits,
)
from orderbias.training.losses import classification_loss, ranking_loss
from orderbias.training.train import build_optimizer, build_scheduler, evaluate_validation, train_judge


def tokenizer():
    vocab = {token: i for i, token in enumerate([
        "<unk>", "<pad>", "<s>", "</s>", "alpha", "beta", "good", "bad", "request", "other", "A", "B",
    ])}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    result = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>",
                                    bos_token="<s>", eos_token="</s>")
    result.chat_template = "{% for m in messages %}<s>{{ m['role'] }}\n{{ m['content'] }}</s>{% endfor %}"
    return result


def model():
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


def config(kind="pairwise", augmentation="none"):
    return ExperimentConfig.from_dict({
        "backbone": {"compute_dtype": "float32", "max_input_length": 256},
        "judge": {"kind": kind},
        "data": {"augmentation": augmentation, "prompt_max_tokens": 8, "answer_max_tokens": 8},
        "training": {"max_steps": 1, "eval_steps": 1, "logging_steps": 1,
                     "batch_size": 2, "gradient_accumulation_steps": 2},
    })


def example(i=0, split="train", label=0):
    return UnifiedExample(str(i), "request" if split == "train" else "other request", "good alpha", "bad beta",
                          label, "helpsteer2", split, "preference", margin=2, prompt_group=f"{split}-group")


def test_swap_flips_labels_and_preserves_truncated_text():
    cfg = config(augmentation="both_orders")
    cfg = replace(cfg, data=replace(cfg.data, answer_max_tokens=2))
    enc = JudgeEncoder(tokenizer(), cfg)
    ex = replace(example(label=1), response_a="alpha beta good bad", response_b="bad good beta alpha")
    row = enc.encode(ex, include_reversed=True)
    assert row.canonical.label == 1 and row.reversed.label == 0
    assert row.truncated == (False, True, True)
    rendered = enc.render("request", "alpha beta", "bad good")
    assert row.canonical.sequences[0] == tuple(enc.tokenizer.encode(rendered, add_special_tokens=False))
    assert row.reversed.sequences[0] == tuple(enc.tokenizer.encode(enc.render("request", "bad good", "alpha beta"), add_special_tokens=False))
    assert ex.response_a == "alpha beta good bad" and ex.label == 1


def test_independent_scorer_never_receives_competitor_or_slot():
    enc = JudgeEncoder(tokenizer(), config("shared_scorer"))
    row = enc.encode(example())
    changed = enc.encode(replace(example(), response_b="beta beta beta"))
    assert row.canonical.sequences[0] == changed.canonical.sequences[0]
    assert row.reversed is None
    text = enc.render("request", "good alpha")
    assert "Answer A:" not in text and "Answer B:" not in text and "bad beta" not in text
    inputs, labels = collate_views([row.canonical, changed.canonical], enc.tokenizer.pad_token_id)
    assert inputs.input_ids.shape[0] == 4
    assert labels.tolist() == [0, 0]
    assert inputs.attention_mask.sum(1).tolist() == [len(seq) for view in [row.canonical, changed.canonical] for seq in view.sequences]


def test_template_overflow_fails_instead_of_slot_truncation():
    cfg = config()
    cfg = replace(cfg, backbone=replace(cfg.backbone, max_input_length=32))
    with pytest.raises(ValueError, match="never truncated"):
        JudgeEncoder(tokenizer(), cfg).encode(example())


def test_plain_template_and_unicode_safe_independent_truncation():
    tok = tokenizer()
    assert truncate_text("alpha beta good", 2, tok) == ("alpha beta", True, 3)
    assert truncate_text("你好 alpha beta", 1, tok)[0] == "你好"
    cfg = config()
    cfg = replace(cfg, backbone=replace(cfg.backbone, use_chat_template=False))
    text = JudgeEncoder(tok, cfg).render("request", "good", "bad")
    assert "<s>" not in text and "Answer A:\ngood" in text


def test_split_validation_rejects_leaks_external_eval_and_duplicates(tmp_path):
    train, val = [example()], [example(1, "validation")]
    validate_training_splits(train, val)
    for invalid in ([replace(val[0], prompt="request")],
                    [replace(val[0], prompt_group="train-group")],
                    [replace(val[0], source="llmbar")],
                    [replace(val[0], example_id="0")]):
        with pytest.raises(ValueError):
            validate_training_splits(train, invalid)
    with pytest.raises(ValueError):
        validate_training_splits([], val)
    path = tmp_path / "unified.jsonl"
    path.write_text(json.dumps(train[0].to_dict()) + "\n", encoding="utf-8")
    assert read_examples(path) == train
    path.write_text("{}\n")
    with pytest.raises(ValueError, match=":1"):
        read_examples(path)


def test_augmented_stream_covers_both_orders_before_next_pair():
    enc = JudgeEncoder(tokenizer(), config())
    rows = [enc.encode(example(i), include_reversed=True) for i in range(3)]
    items = list(islice(presentation_stream(rows, both_orders=True, seed=7), 12))
    assert items == list(islice(presentation_stream(rows, both_orders=True, seed=7), 12))
    for first, second in zip(items[::2], items[1::2]):
        assert first[0] == second[0]
        assert {first[1].swapped, second[1].swapped} == {False, True}
        assert first[1].label == 1 - second[1].label
    standard = list(islice(presentation_stream(rows, both_orders=False, seed=7), 6))
    assert [i for i, _ in standard] == [i for i, _ in items[::2]]


def test_losses_preference_sign_swap_equivalence_and_extreme_stability():
    labels = torch.tensor([0, 1])
    a = torch.tensor([3.0, -3.0], requires_grad=True)
    b = torch.tensor([-2.0, 2.0], requires_grad=True)
    good = ranking_loss(a, b, labels)
    assert good < ranking_loss(b, a, labels)
    torch.testing.assert_close(good, ranking_loss(b, a, 1 - labels))
    logits = torch.stack([a, b], dim=1)
    torch.testing.assert_close(good, classification_loss(logits, labels))
    good.backward()
    assert a.grad[0] < 0 and a.grad[1] > 0
    assert b.grad[0] > 0 and b.grad[1] < 0
    assert torch.isfinite(ranking_loss(torch.tensor([-1e6]), torch.tensor([1e6]), torch.tensor([0])))
    with pytest.raises(ValueError):
        ranking_loss(a, b, torch.tensor([2, 0]))
    with pytest.raises(ValueError):
        classification_loss(torch.ones(2, 1), labels)


def test_tokenizer_load_uses_existing_padding_and_saves_template(tmp_path):
    tok = tokenizer()
    tok.pad_token = None
    tok.save_pretrained(tmp_path)
    cfg = config()
    cfg = replace(cfg, backbone=replace(cfg.backbone, checkpoint=str(tmp_path)))
    loaded, manifest = load_tokenizer(cfg, local_files_only=True)
    assert loaded.pad_token_id == loaded.eos_token_id
    assert manifest["enable_thinking"] is False and manifest["add_generation_prompt"] is False
    assert len(manifest["chat_template_sha256"]) == 64


@pytest.mark.parametrize("kind,augmentation", [("pairwise", "none"), ("pairwise", "both_orders"), ("shared_scorer", "none")])
def test_actual_training_all_conditions_reload_and_exposure(kind, augmentation, tmp_path):
    cfg = config(kind, augmentation)
    base = model()
    base_dir = tmp_path / "base"
    base.save_pretrained(base_dir)
    cfg = replace(cfg, backbone=replace(cfg.backbone, checkpoint=str(base_dir)))
    original = [(p, p.detach().clone()) for p in base.parameters()]
    judge = build_judge(prepare_backbone(base, cfg), cfg)
    head_before = judge.head.weight.detach().clone()
    adapter_before = {name: p.detach().clone() for name, p in judge.backbone.named_parameters() if p.requires_grad}
    train, val = [example(0), example(1, label=1)], [example(2, "validation")]
    tok = tokenizer()
    output = tmp_path / "run"
    summary = train_judge(judge, tok, train, val, output)
    assert summary["training_presentations"] == 4
    assert summary["reversed_presentations"] == (2 if augmentation == "both_orders" else 0)
    assert summary["optimizer_steps"] == 1 and summary["training_input_tokens"] > 0
    assert summary["best_step"] == 1 and summary["best_validation_loss"] >= 0
    assert not torch.equal(head_before, judge.head.weight)
    assert any(not torch.equal(adapter_before[name], p) for name, p in judge.backbone.named_parameters() if p.requires_grad)
    assert all(torch.equal(p, initial) for p, initial in original)
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert metrics[0]["validation"]["presentations"] == (2 if kind == "pairwise" else 1)
    saved = output / summary["final_checkpoint"]
    restored = load_judge_checkpoint(saved, local_files_only=True)
    restored_tok = AutoTokenizer.from_pretrained(saved / "tokenizer", local_files_only=True)
    encoded = [JudgeEncoder(restored_tok, cfg).encode(ex, include_reversed=kind == "pairwise") for ex in val]
    before = evaluate_validation(judge, encoded, tok)
    after = evaluate_validation(restored, encoded, restored_tok)
    assert before == after
    assert (output / "input_preview.json").is_file()
    with pytest.raises(FileExistsError):
        train_judge(judge, tok, train, val, output)


def test_scheduler_and_optimizer_groups():
    cfg = config()
    judge = build_judge(prepare_backbone(model(), cfg), cfg)
    optimizer = build_optimizer(judge)
    assert {group["name"] for group in optimizer.param_groups} == {"head", "adapters"}
    settings = replace(cfg.training, max_steps=4, warmup_ratio=0.5)
    scheduler = build_scheduler(optimizer, settings)
    assert optimizer.param_groups[0]["lr"] == 0
    for _ in range(2):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == settings.head_learning_rate
    for _ in range(2):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0


def test_head_only_gradient_accumulation_matches_one_full_batch(tmp_path):
    base_dir = tmp_path / "base"
    model().save_pretrained(base_dir)
    cfg = config()
    cfg = replace(cfg, backbone=replace(cfg.backbone, checkpoint=str(base_dir)),
                  lora=replace(cfg.lora, enabled=False))
    full_cfg = replace(cfg, training=replace(cfg.training, batch_size=4, gradient_accumulation_steps=1))
    accumulated, full = load_judge(cfg, local_files_only=True), load_judge(full_cfg, local_files_only=True)
    train, val = [example(), replace(example(1), response_a="bad alpha", response_b="good beta", label=1)], [example(2, "validation")]
    tok = tokenizer()
    train_judge(accumulated, tok, train, val, tmp_path / "accumulated")
    train_judge(full, tok, train, val, tmp_path / "full")
    torch.testing.assert_close(accumulated.head.weight, full.head.weight, rtol=1e-5, atol=1e-7)


def test_training_cli_local_checkpoint(tmp_path, monkeypatch):
    base_dir = tmp_path / "base"
    model().save_pretrained(base_dir)
    tokenizer().save_pretrained(base_dir)
    cfg = config()
    cfg = replace(cfg, backbone=replace(cfg.backbone, checkpoint=str(base_dir)))
    model_config = tmp_path / "model.json"
    cfg.save_json(model_config)
    empty_config = tmp_path / "experiment.json"
    empty_config.write_text("{}")
    train_file, val_file = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    train_file.write_text(json.dumps(example().to_dict()) + "\n")
    val_file.write_text(json.dumps(example(1, "validation").to_dict()) + "\n")
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(sys, "argv", [
        "train_judge.py", "--model-config", str(model_config), "--experiment-config", str(empty_config),
        "--train-data", str(train_file), "--validation-data", str(val_file),
        "--output-dir", str(tmp_path / "run"), "--local-files-only",
    ])
    runpy.run_path(str(root / "scripts/train_judge.py"), run_name="__main__")
    assert json.loads((tmp_path / "run/summary.json").read_text())["status"] == "complete"
