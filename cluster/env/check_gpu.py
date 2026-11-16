"""Offline CUDA/BF16 forward/backward smoke check using tiny actual Qwen3.5 judges."""
import json
import platform
from importlib.metadata import version

import torch
from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

from orderbias.models.backbone import prepare_backbone
from orderbias.models.config import ExperimentConfig
from orderbias.models.interfaces import TokenBatch
from orderbias.models.judges import build_judge


def main():
    assert torch.cuda.is_available(), "Run this check inside a Slurm GPU allocation"
    assert torch.cuda.is_bf16_supported(), "Default training configs require BF16"
    print(json.dumps({
        "python": platform.python_version(), "gpu": torch.cuda.get_device_name(0),
        "cuda_runtime": torch.version.cuda,
        "versions": {name: version(name) for name in ("torch", "transformers", "peft", "accelerate")},
    }), flush=True)
    for kind in ("pairwise", "shared_scorer"):
        cfg = ExperimentConfig.from_dict({
            "backbone": {"compute_dtype": "bfloat16"}, "judge": {"kind": kind},
        })
        model = Qwen3_5TextModel(Qwen3_5TextConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            head_dim=16, layer_types=["linear_attention", "full_attention"],
            linear_num_key_heads=2, linear_num_value_heads=2,
            linear_key_head_dim=8, linear_value_head_dim=8,
            max_position_embeddings=4096,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                             "partial_rotary_factor": 0.5, "mrope_section": [2, 1, 1]},
        )).to(device="cuda:0", dtype=torch.bfloat16)
        judge = build_judge(prepare_backbone(model, cfg), cfg)
        ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], device="cuda:0")
        output = judge(TokenBatch(ids, torch.ones_like(ids)))
        values = output.logits if kind == "pairwise" else output.scores
        assert torch.isfinite(values).all()
        values.float().square().mean().backward()
        grads = [(name, p.grad) for name, p in judge.named_parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for _, g in grads)
        torch.optim.AdamW([p for p in judge.parameters() if p.requires_grad], lr=1e-4).step()
        torch.cuda.synchronize()
        print(f"{kind}: BF16 CUDA forward/backward/optimizer PASS", flush=True)


if __name__ == "__main__":
    main()
