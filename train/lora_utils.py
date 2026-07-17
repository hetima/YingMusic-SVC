"""YingMusic-SVC向けLoRA層の注入・保存・マージ処理。"""

from pathlib import Path
from typing import Iterable

import loralib as lora
import torch
from torch import nn


DEFAULT_TARGET_MODULES = ("wqkv", "wq", "wkv", "wo", "w1", "w2", "w3")


def inject_lora(
    model: nn.Module,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: Iterable[str] = DEFAULT_TARGET_MODULES,
) -> list[str]:
    """DiT Transformerの対象Linear層をLoRA層へ置き換える。"""
    targets = set(target_modules)
    replaced = []
    for module_name, module in list(model.named_modules()):
        if not module_name.startswith("estimator.transformer.layers."):
            continue
        child_name = module_name.rsplit(".", 1)[-1]
        if child_name not in targets or not isinstance(module, nn.Linear):
            continue

        parent_name, attribute_name = module_name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        replacement = lora.Linear(
            module.in_features,
            module.out_features,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            merge_weights=False,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        replacement.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            replacement.bias.data.copy_(module.bias.data)
        setattr(parent, attribute_name, replacement)
        replaced.append(module_name)

    if not replaced:
        raise RuntimeError("LoRAの対象となるDiT Linear層が見つかりませんでした。")
    return replaced


def mark_only_lora_trainable(model: nn.Module) -> None:
    """LoRAパラメータ以外をすべて凍結する。"""
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora_" in name


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """LoRA差分だけをCPU上のstate dictとして返す。"""
    return {
        key: value.detach().cpu()
        for key, value in lora.lora_state_dict(model, bias="none").items()
    }


def load_lora_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """注入済みモデルへLoRA差分を読み込む。"""
    result = model.load_state_dict(state, strict=False)
    expected = set(lora.lora_state_dict(model, bias="none"))
    missing = sorted(expected - set(state))
    unexpected = [key for key in result.unexpected_keys if "lora_" in key]
    if missing:
        raise RuntimeError(f"不足しているLoRAパラメータがあります: {missing}")
    if unexpected:
        raise RuntimeError(f"読み込めないLoRAパラメータがあります: {unexpected}")


def merge_lora_modules(model: nn.Module, scale: float = 1.0) -> int:
    """LoRAを通常Linearへマージし、LoRA依存のないモデル構造へ戻す。"""
    merged = 0
    for module_name, module in list(model.named_modules()):
        if not isinstance(module, lora.Linear) or module.r <= 0:
            continue
        parent_name, attribute_name = module_name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        weight = module.weight.data + (
            module.lora_B.data @ module.lora_A.data
        ) * module.scaling * scale
        replacement = nn.Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        replacement.weight.data.copy_(weight)
        if module.bias is not None:
            replacement.bias.data.copy_(module.bias.data)
        setattr(parent, attribute_name, replacement)
        merged += 1
    return merged


def checkpoint_step(path: str) -> int:
    """LoRA中間checkpoint名からstep番号を取得する。"""
    return int(Path(path).stem.rsplit("_step_", 1)[1])
