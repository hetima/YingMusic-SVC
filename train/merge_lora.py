"""LoRA差分をベースモデルへマージして従来形式で保存するCLI。"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

from modules.commons import build_model, load_checkpoint, recursive_munch
from train.lora_utils import (
    detect_checkpoint_type,
    inject_lora,
    load_lora_state,
    merge_lora_modules,
)


def main(args):
    """ベースcheckpointとLoRA差分からマージ済みモデルを生成する。"""
    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    model = build_model(recursive_munch(config["model_params"]), stage="DiT")
    model, _, _, _ = load_checkpoint(
        model,
        None,
        args.base,
        load_only_params=True,
        ignore_modules=[],
        is_distributed=False,
    )

    state = torch.load(args.lora, map_location="cpu", weights_only=False)
    if detect_checkpoint_type(state) != "lora":
        raise ValueError(f"LoRA checkpointの形式が不正です: {args.lora}")
    metadata = state["metadata"]
    inject_lora(
        model.cfm,
        rank=metadata["rank"],
        alpha=metadata["alpha"],
        dropout=metadata.get("dropout", 0.0),
        target_modules=metadata["target_modules"],
    )
    load_lora_state(model.cfm, state["lora"])
    merged_count = merge_lora_modules(model.cfm, scale=args.scale)

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    output = {"net": {key: model[key].state_dict() for key in model}}
    if "spk_embedding" in state:
        output["spk_embedding"] = state["spk_embedding"]
    torch.save(output, args.output)
    print(f"Merged {merged_count} LoRA modules into {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--lora", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    main(parser.parse_args())
