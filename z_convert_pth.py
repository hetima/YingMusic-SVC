import argparse
from pathlib import Path

import torch


def convert_checkpoint(input_path: Path, output_path: Path) -> list[str]:
    """学習途中のチェックポイントを推論用の最終保存形式へ変換する。"""
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("チェックポイントの最上位が辞書ではありません。")
    if "net" not in checkpoint or not isinstance(checkpoint["net"], dict):
        raise ValueError("チェックポイントに辞書形式の 'net' がありません。")

    converted = {"net": checkpoint["net"]}
    if "spk_embedding" in checkpoint:
        converted["spk_embedding"] = checkpoint["spk_embedding"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, output_path)
    return list(converted)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="学習途中の.pthを最終保存形式へ変換します。"
    )
    parser.add_argument("input", type=Path, help="変換元の.pthファイル")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="変換先（省略時: <入力名>_final.pth）",
    )
    args = parser.parse_args()

    input_path = args.input
    if not input_path.is_file():
        parser.error(f"入力ファイルが見つかりません: {input_path}")

    output_path = args.output or input_path.with_name(
        f"{input_path.stem}_final{input_path.suffix}"
    )
    if input_path.resolve() == output_path.resolve():
        parser.error("入力ファイルと出力ファイルには別のパスを指定してください。")

    saved_keys = convert_checkpoint(input_path, output_path)
    print(f"変換しました: {output_path}")
    print(f"保存したキー: {', '.join(saved_keys)}")


if __name__ == "__main__":
    main()
