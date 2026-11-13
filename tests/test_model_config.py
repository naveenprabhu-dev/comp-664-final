"""Offline configuration boundaries: no checkpoint or GPU dependencies."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from orderbias.models.config import ExperimentConfig, LoRAConfig, load_config

ROOT = Path(__file__).resolve().parents[1]


def load(model="qwen35_08b", experiment="pairwise", **overrides):
    return load_config(
        ROOT / "configs/models" / f"{model}.yaml",
        ROOT / "configs/experiments" / f"{experiment}.yaml",
        overrides=overrides,
    )


@pytest.mark.parametrize("model", ["qwen35_08b", "qwen35_2b"])
@pytest.mark.parametrize("experiment", ["pairwise", "pairwise_swap", "shared_scorer"])
def test_shipped_combinations(model, experiment):
    config = load(model, experiment)
    assert config.judge.output_dim == (1 if experiment == "shared_scorer" else 2)
    assert config.data.augmentation == ("both_orders" if experiment == "pairwise_swap" else "none")


def test_modes_explicit_targets_and_roundtrip(tmp_path):
    config = load(
        lora={"target_preset": None, "target_modules": ["layers.0.projection"]},
        quantization={"mode": "4bit"},
    )
    assert config.lora.target_modules == ("layers.0.projection",)
    path = tmp_path / "resolved.json"
    config.save_json(path)
    assert ExperimentConfig.from_dict(json.loads(path.read_text())) == config
    assert not load(lora={"enabled": False}).lora.enabled


def test_merge_precedence_and_no_mutation(tmp_path):
    model = tmp_path / "model.yaml"
    experiment = tmp_path / "experiment.yaml"
    model.write_text("lora: {rank: 16, dropout: 0.1}\n")
    experiment.write_text("lora: {rank: 32}\n")
    override = {"lora": {"alpha": 64}}
    config = load_config(model, experiment, overrides=override)
    assert (config.lora.rank, config.lora.alpha, config.lora.dropout) == (32, 64, 0.1)
    assert override == {"lora": {"alpha": 64}}
    assert load_config(model, experiment, overrides={"lora": {"rank": 4}}).lora.rank == 4


@pytest.mark.parametrize("override", [
    {"lora": {"rnak": 8}},
    {"typo": {}},
    {"lora": {"rank": True}},
    {"lora": {"rank": "8"}},
    {"lora": {"rank": 0}},
    {"lora": {"dropout": 1}},
    {"lora": {"alpha": float("nan")}},
    {"lora": {"target_modules": ["projection"]}},
    {"lora": {"target_preset": None}},
    {"lora": {"target_preset": None, "target_modules": ["p", "p"]}},
    {"lora": {"target_preset": None, "target_modules": ["p"], "exclude_modules": ["p"]}},
    {"judge": {"kind": "shared_scorer"}, "data": {"augmentation": "both_orders"}},
    {"backbone": {"max_input_length": 3584}},
    {"quantization": {"mode": "8bit"}},
    {"quantization": {"mode": "4bit", "compute_dtype": "float16"}},
    {"training": {"max_steps": -1}},
    {"training": {"warmup_ratio": 1}},
    {"training": {"eval_steps": 0}},
    {"training": {"max_grad_norm": 0}},
    {"data": {"augmentation": "both_orders"},
     "training": {"batch_size": 1, "gradient_accumulation_steps": 1, "max_steps": 1}},
    {"backbone": {"enable_thinking": True}},
])
def test_invalid_settings_fail(override):
    with pytest.raises(ValueError):
        load(**override)


def test_direct_construction_validates():
    with pytest.raises(ValueError, match="positive"):
        LoRAConfig(rank=-1)


def test_partial_files_validate_after_merging(tmp_path):
    model, experiment = tmp_path / "model.yaml", tmp_path / "experiment.yaml"
    model.write_text("backbone: {compute_dtype: float16}\n")
    experiment.write_text("quantization: {mode: 4bit, compute_dtype: float16}\n")
    assert load_config(model, experiment).quantization.compute_dtype == "float16"
    model.write_text("lora: {rnak: 16}\n")
    with pytest.raises(ValueError, match="unknown"):
        load_config(model, experiment, overrides={"lora": {"rnak": 8}})


def test_non_mapping_yaml_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- not a mapping\n")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path, path)


def test_lightweight_imports_without_training_stack():
    code = '''
import sys
class BlockHeavy:
    def find_spec(self, fullname, *args):
        if fullname.split(".")[0] in {"torch", "transformers", "peft", "bitsandbytes", "trl"}:
            raise AssertionError("heavy import: " + fullname)
sys.meta_path.insert(0, BlockHeavy())
import orderbias
from orderbias.data.schema import UnifiedExample
from orderbias.models.config import ExperimentConfig
from orderbias.models.backbone import BackboneReport, load_backbone
from orderbias.training.data import JudgeEncoder, validate_training_splits
from orderbias.models.interfaces import TokenBatch, PairwiseOutput, ScorerOutput
assert ExperimentConfig().judge.output_dim == 2
'''
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    subprocess.run([sys.executable, "-c", code], env=env, check=True, capture_output=True, text=True)
