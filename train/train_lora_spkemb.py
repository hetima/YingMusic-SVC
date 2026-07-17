"""train_yingmusic_ft_spkemb.pyを基にしたLoRA学習スクリプト。"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch.multiprocessing as mp

from train.lora_spkemb_common import (
    LoRASpkEmbTrainerMixin,
    build_parser,
    run_training,
)
from train_yingmusic_ft_spkemb import Trainer as FullTrainer


class Trainer(LoRASpkEmbTrainerMixin, FullTrainer):
    """通常spk_embedding版のLoRA Trainer。"""

    trainer_variant = "spkemb"


if __name__ == "__main__":
    if sys.platform == "win32":
        mp.freeze_support()
        mp.set_start_method("spawn", force=True)
    parser = build_parser("spk_embedding版のLoRA学習を実行します。")
    run_training(Trainer, parser.parse_args())
