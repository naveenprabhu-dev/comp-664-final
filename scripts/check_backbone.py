"""Explicit checkpoint load/gradient smoke check for CPU or a cluster GPU.

No tokenization or task head: synthetic token IDs exercise the full text stack.
Weights may be downloaded unless --local-files-only is supplied. Example:
  python scripts/check_backbone.py --device cuda:0 --quantization 4bit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orderbias.models.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default="configs/models/qwen35_08b.yaml")
    parser.add_argument("--experiment-config", default="configs/experiments/pairwise.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--quantization", choices=["none", "4bit"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/backbone_check"))
    args = parser.parse_args()
    overrides = {}
    dtype = args.dtype or ("float32" if args.device == "cpu" else None)
    if dtype:
        overrides["backbone"] = {"compute_dtype": dtype}
        overrides["quantization"] = {"compute_dtype": dtype}
    if args.quantization:
        overrides.setdefault("quantization", {})["mode"] = args.quantization
    config = load_config(args.model_config, args.experiment_config, overrides=overrides)

    import torch
    from orderbias.models.backbone import load_backbone

    torch.manual_seed(config.training.seed)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    loaded = load_backbone(config, device=args.device, local_files_only=args.local_files_only)
    device = next(loaded.model.parameters()).device
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device=device)
    output = loaded.model(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state
    if not torch.isfinite(output).all():
        raise RuntimeError("Non-finite backbone output")
    if config.lora.enabled:
        output[:, -1, 0].sum().backward()
        trainable = [p for p in loaded.model.parameters() if p.requires_grad]
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable):
            raise RuntimeError("Missing or non-finite adapter gradients")
        if not any(p.grad.abs().sum() > 0 for p in trainable):
            raise RuntimeError("All adapter gradients are zero")
    if any(p.grad is not None for p in loaded.model.parameters() if not p.requires_grad):
        raise RuntimeError("Frozen parameter received a gradient")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config.save_json(args.output_dir / "resolved_config.json")
    loaded.report.save_json(args.output_dir / "backbone_report.json")
    summary = {
        "status": "passed",
        "output_shape": list(output.shape),
        "adapter_modules": len(loaded.report.adapters),
        "trainable_parameters": loaded.report.trainable_parameters,
        "total_parameters": loaded.report.total_parameters,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }
    (args.output_dir / "smoke_result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
