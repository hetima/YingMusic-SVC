"""train_yingmusic_ft_cosine.pyを基にしたLoRA学習スクリプト。"""

import argparse
import glob
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.multiprocessing as mp
from torch.optim import AdamW
from tqdm import tqdm

from optimizers_cosine import CosineWarmupScheduler
from train.lora_utils import (
    DEFAULT_TARGET_MODULES,
    checkpoint_step,
    inject_lora,
    load_lora_state,
    lora_state_dict,
    mark_only_lora_trainable,
)
from train_yingmusic_ft_cosine import Trainer as FullTrainer


class LoRAOptimizerAdapter:
    """既存TrainerのMultiOptimizer呼び出しを単一LoRA optimizerへ接続する。"""

    def __init__(self, optimizer, scheduler):
        self.optimizer = optimizer
        self.scheduler_object = scheduler

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self, key=None, scaler=None):
        if key != "cfm":
            return
        if scaler is None:
            self.optimizer.step()
        else:
            scaler.step(self.optimizer)

    def scheduler(self, *args, key=None):
        if key == "cfm":
            self.scheduler_object.step()


class Trainer(FullTrainer):
    """DiTのLoRAパラメータだけを更新するTrainer。"""

    def __init__(
        self,
        *args,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_lr=1e-4,
        **kwargs,
    ):
        pretrained_ckpt_path = kwargs.get("pretrained_ckpt_path")
        if not pretrained_ckpt_path:
            raise ValueError("LoRA学習には--pretrained-ckptが必要です。")

        super().__init__(*args, **kwargs)
        self.base_checkpoint = os.path.abspath(pretrained_ckpt_path)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_lr = lora_lr
        self.target_modules = list(DEFAULT_TARGET_MODULES)

        self.replaced_modules = inject_lora(
            self.model.cfm,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=self.target_modules,
        )
        mark_only_lora_trainable(self.model.cfm)
        trainable = [p for p in self.model.cfm.parameters() if p.requires_grad]
        optimizer = AdamW(
            trainable,
            lr=lora_lr,
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=0.01,
        )
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_steps=500,
            total_steps=self.max_steps,
            lr_min=1e-6,
        )
        self.optimizer = LoRAOptimizerAdapter(optimizer, scheduler)
        self._resume_lora_if_available()
        trainable_count = sum(p.numel() for p in trainable)
        print(f"LoRA modules: {len(self.replaced_modules)}, trainable parameters: {trainable_count:,}")

    def _metadata(self):
        """保存・互換性確認に使うLoRA設定を返す。"""
        return {
            "format": "yingmusic_lora_v1",
            "base_checkpoint": self.base_checkpoint,
            "rank": self.lora_rank,
            "alpha": self.lora_alpha,
            "dropout": self.lora_dropout,
            "target_modules": self.target_modules,
        }

    def _resume_lora_if_available(self):
        """runディレクトリ内の最新LoRA checkpointから学習を再開する。"""
        checkpoints = glob.glob(os.path.join(self.log_dir, "LoRA_epoch_*_step_*.pth"))
        if not checkpoints:
            self.epoch, self.iters = 0, 0
            return
        latest = max(checkpoints, key=checkpoint_step)
        state = torch.load(latest, map_location="cpu", weights_only=False)
        metadata = state.get("metadata", {})
        expected = self._metadata()
        for key in ("format", "rank", "alpha", "target_modules"):
            if metadata.get(key) != expected[key]:
                raise ValueError(
                    f"LoRA checkpointの{key}が現在の設定と一致しません: "
                    f"{metadata.get(key)!r} != {expected[key]!r}"
                )
        load_lora_state(self.model.cfm, state["lora"])
        self.optimizer.optimizer.load_state_dict(state["optimizer"])
        self.optimizer.scheduler_object.load_state_dict(state["scheduler"])
        self.epoch = state.get("epoch", 0)
        self.iters = state.get("iters", 0)
        print(f"Resumed LoRA checkpoint from {latest} at step {self.iters}")

    def _save_checkpoint(self):
        """学習再開用のLoRA中間checkpointを保存する。"""
        state = {
            "metadata": self._metadata(),
            "lora": lora_state_dict(self.model.cfm),
            "optimizer": self.optimizer.optimizer.state_dict(),
            "scheduler": self.optimizer.scheduler_object.state_dict(),
            "iters": self.iters,
            "epoch": self.epoch,
        }
        path = os.path.join(
            self.log_dir,
            f"LoRA_epoch_{self.epoch:05d}_step_{self.iters:05d}.pth",
        )
        torch.save(state, path)
        print(f"LoRA checkpoint saved at {path}")

    def train_one_epoch(self):
        """1 epoch学習し、指定間隔でLoRA checkpointを保存する。"""
        _ = [self.model[key].train() for key in self.model]
        for batch in tqdm(self.train_dataloader):
            batch = [value.to(self.device) for value in batch]
            loss = self.train_one_step(batch)
            self.ema_loss = (
                self.ema_loss * self.loss_smoothing_rate + loss * (1 - self.loss_smoothing_rate)
                if self.iters > 0 else loss
            )
            if self.iters % self.log_interval == 0:
                print(f"epoch {self.epoch}, step {self.iters}, loss: {self.ema_loss}")
            self.iters += 1
            if self.iters % self.save_interval == 0:
                self._save_checkpoint()
            if self.iters >= self.max_steps:
                break

    def train(self):
        """LoRA学習を実行し、最終差分のみを保存する。"""
        self.ema_loss = 0
        self.loss_smoothing_rate = 0.99
        start_epoch = self.epoch
        for epoch in range(start_epoch, self.n_epochs):
            self.epoch = epoch
            self.train_one_epoch()
            if self.iters >= self.max_steps:
                break

        state = {
            "metadata": self._metadata(),
            "lora": lora_state_dict(self.model.cfm),
        }
        path = os.path.join(self.log_dir, "lora_final.pth")
        torch.save(state, path)
        print(f"Final LoRA saved at {path}")


def main(args):
    """CLI引数からLoRA Trainerを構築して学習する。"""
    trainer = Trainer(
        config_path=args.config,
        pretrained_ckpt_path=args.pretrained_ckpt,
        data_dir=args.dataset_dir,
        run_name=args.run_name,
        batch_size=args.batch_size,
        steps=args.max_steps,
        max_epochs=args.max_epochs,
        save_interval=args.save_every,
        num_workers=args.num_workers,
        device=args.device,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_lr=args.lora_lr,
    )
    trainer.train()


if __name__ == "__main__":
    if sys.platform == "win32":
        mp.freeze_support()
        mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained-ckpt", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--run-name", default="yingmusic_lora")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if torch.backends.mps.is_available():
        args.device = "mps"
    else:
        args.device = f"cuda:{args.gpu}"
    main(args)
