import argparse
from pathlib import Path

import torch

from train.lora_utils import detect_checkpoint_type


def convert_checkpoint(
    input_path: Path,
    output_path: Path,
    checkpoint: dict | None = None,
) -> list[str]:
    """通常またはLoRAの中間checkpointを最終保存形式へ変換する。"""
    if checkpoint is None:
        checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("チェックポイントの最上位が辞書ではありません。")

    checkpoint_type = detect_checkpoint_type(checkpoint)
    if checkpoint_type == "model":
        converted = {"net": checkpoint["net"]}
        if "spk_embedding" in checkpoint:
            converted["spk_embedding"] = checkpoint["spk_embedding"]
    elif checkpoint_type == "lora":
        converted = {
            "metadata": checkpoint["metadata"],
            "lora": checkpoint["lora"],
        }
        if "spk_embedding" in checkpoint:
            converted["spk_embedding"] = checkpoint["spk_embedding"]
    else:
        raise ValueError(
            "チェックポイントに辞書形式の 'net'、または "
            "'metadata' と 'lora' がありません。"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, output_path)
    return list(converted)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="通常またはLoRAの学習途中の.pthを最終保存形式へ変換します。"
    )
    parser.add_argument("input", type=Path, help="変換元の.pthファイル")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="変換先（省略時: 通常は*_final.pth、LoRAは*_final.lora.pth）",
    )
    args = parser.parse_args()

    input_path = args.input
    if not input_path.is_file():
        parser.error(f"入力ファイルが見つかりません: {input_path}")

    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    checkpoint_type = detect_checkpoint_type(checkpoint)
    if args.output is not None:
        output_path = args.output
    elif checkpoint_type == "lora":
        base_stem = input_path.name.removesuffix(".lora.pth").removesuffix(".pth")
        output_path = input_path.with_name(f"{base_stem}_final.lora.pth")
    else:
        output_path = input_path.with_name(
            f"{input_path.stem}_final{input_path.suffix}"
        )
    if input_path.resolve() == output_path.resolve():
        parser.error("入力ファイルと出力ファイルには別のパスを指定してください。")

    saved_keys = convert_checkpoint(input_path, output_path, checkpoint=checkpoint)
    print(f"変換しました: {output_path}")
    print(f"保存したキー: {', '.join(saved_keys)}")


if __name__ == "__main__":
    main()
