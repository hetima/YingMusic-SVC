"""_train_spkemb_v2.pyを基にしたLoRA学習スクリプト。"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch.multiprocessing as mp

from _train_spkemb_v2 import (
    SCHEDULER_LR_MIN,
    SCHEDULER_WARMUP,
    Trainer as FullTrainer,
)
from train.lora_spkemb_common import (
    LoRASpkEmbTrainerMixin,
    build_parser,
    run_training,
)


class Trainer(LoRASpkEmbTrainerMixin, FullTrainer):
    """v2 spk_embedding版のLoRA Trainer。"""

    trainer_variant = "spkemb_v2"
    scheduler_warmup = SCHEDULER_WARMUP
    scheduler_lr_min = SCHEDULER_LR_MIN


if __name__ == "__main__":
    if sys.platform == "win32":
        mp.freeze_support()
        mp.set_start_method("spawn", force=True)
    parser = build_parser("spk_embedding v2版のLoRA学習を実行します。")
    run_training(Trainer, parser.parse_args())
