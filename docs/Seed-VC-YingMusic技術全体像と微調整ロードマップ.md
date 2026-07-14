# Seed-VC / YingMusic-SVC 技術全体像と微調整ロードマップ

> 複数回の対話と Grok での検討内容を整理 | 2026-05

---

## 目次

1. [プロジェクト背景と学習履歴](#1-プロジェクト背景と学習履歴)
2. [Seed-VC V1 アーキテクチャ完全解析](#2-seed-vc-v1-アーキテクチャ完全解析)
3. [V1 学習フローの段階別分解](#3-v1-学習フローの段階別分解)
4. [Seed-VC V2 アーキテクチャと AR 失敗分析](#4-seed-vc-v2-アーキテクチャと-ar-失敗分析)
5. [V2 推論における target 参照音声の役割](#5-v2-推論における-target-参照音声の役割)
6. [YingMusic-SVC vs V1 モジュール別比較](#6-yingmusic-svc-vs-v1-モジュール別比較)
7. [RVC Timbre Shifter の現状と代替案](#7-rvc-timbre-shifter-の現状と代替案)
8. [YingMusic 微調整の実行ルート](#8-yingmusic-微調整の実行ルート)

---

## 1. プロジェクト背景と学習履歴

### 目的
natori_sana の話声/歌声を花丸（hanamaru）の音色へ変換し、花丸の話し方のスタイル/歌唱表現力を保持する。

### 既存の学習履歴

| 学習名 | モデル版 | データセット | ステップ数 | lr | 状態 |
|------|:---:|------|:--:|:--:|:--:|
| 花丸PRUE-small | V1 | ~50 件のドライボーカル | 1750 | 1e-5 | 初期テスト |
| my_run | V1 | 少量のテストデータ | 1750 | 1e-5 | テスト |
| hanamaru_full_fav | V1 | 1153 件（93曲を分割） | 5500 | 1e-5 | 正式 V1 |
| hanamaru_full_fav_step15000 | V1 | 同上 | +9500 (to 15000) | 5e-6 | ✅ V1 ベスト |
| hanamaru_v2_cfm | V2 (CFM only) | 花丸ドライボーカル | 6000 | — | CFM 単独学習 |
| hanamaru_v2_cfm_ar | V2 (CFM+AR) | 花丸ドライボーカル | 6000 | — | ❌ AR が話者横断で失敗 |
| yingmusic_exp1 | YingMusic | 1153 件 | 12000 | 5e-6 | ❌ lr が低すぎ、ほぼ学習していない |
| yingmusic_cosine | YingMusic | 1153 件 | 3000 | 1e-4 warmup | ✅ 単体試聴では改善あり |
| RVC | RVC | 1153 件 | 100 epoch | — | ❌ Seed-VC V1 未満 |
| **yingmusic_spkemb_60k** | **YingMusic+spk_emb** | **5603件 / 36.4h** | **60000** | **1e-4 cosine** | **❌ フェイダン状態→保存（独特の厚みあり）** |
| **spkemb_v3_lr3e-5_20k** | **YingMusic+spk_emb_v2** | **5603件 / 36.4h** | **20000** | **3e-5 cosine** | **✅ v3 主力 (warmup 3k)** |

### 3モデル比較（SummerMemory 同一曲）

| モデル | 音色 | 柔らかさ | 清涼感 | 備考 |
|------|:---:|:---:|:---:|------|
| Seed-VC V1 15000歩 | 最も花丸に近い | 刺さる | ⭐⭐⭐ | 均一 L1 loss、高域の制約不足 |
| YingMusic exp1 12000歩 (lr=5e-6) | 柔らかい | 柔らかすぎ→清涼 | ⭐⭐⭐⭐ | lr が低すぎ、重みはほぼ変化なし、CFM cos=0.9997 |
| YingMusic cosine 3000歩 (lr=1e-4) | 改善あり | 柔らかい→清涼 | ⭐⭐⭐⭐⭐ | cosine warmup、CFM cos=0.999999 |
| YingMusic spkemb 60k | — | —→厚い | ⭐⭐ | avg_style 初期化で音色が逸脱、CAMPPlus リアルタイムなら救える |
| **YingMusic v3 20k campplus** | **最も正確** | **中間** | **⭐⭐⭐** | **lr=3e-5, 100 diff, CFM cos=0.996** |

### 現在利用可能なモデル

> 🔗 モデル重みは HF にアップロード済み：[321oll/hanamaru_hareru_YingMusicModel](https://huggingface.co/321oll/hanamaru_hareru_YingMusicModel)。以下のパスは開発環境のローカルパス（参考のみ）。

| モデル | ローカルパス（開発環境） | 用途 |
|------|------|------|
| V1 lr=5e-6 15000歩 | `runs/hanamaru_full_fav_step15000/ft_model.pth` | 歌唱+話声の主力 |
| V2 CFM-only | `runs/hanamaru_v2_cfm/CFM_epoch_00003_step_06000.pth` | V2 話声 CFM |
| YingMusic 事前学習 | `YingMusic-SVC-full.pt` (HF: GiantAILab) | ゼロショット推論 |
| YingMusic exp1 12000歩 | `output_models/yingmusic_exp1/` | **★ 清涼で柔らかい系** |
| YingMusic cosine 3000歩 | `output_models/yingmusic_cosine/` | **★ 最も清涼** |
| **YingMusic spkemb 60k** | **`output_models/yingmusic_spkemb_60k/`** | **保存版（厚いスタイル）** |
| **YingMusic v3 20k** | **`output_models/spkemb_v3_lr3e-5_20k/`** | **✅ 音色が最も正確** |
| hanamaru_avg_style.pt | `output_models/hanamaru_avg_style.pt` | ❌ 平均 CAMPPlus (無効と確認済み) |
| **hanamaru_prompt_style.pt** | **`output_models/hanamaru_prompt_style.pt`** | **単一 CAMPPlus (v3 初期化)** |
| OpenVoice se_db | `modules/openvoice/checkpoints_v2/converter/se_db.pt` | Timbre Shifter 依存 (100004人) |

---

## 9. 学習データセット

### データパイプライン

```
bilibili からダウンロードした .wav (272曲)
  → batch_msst.ps1 (duality_v2 → fused dereverb → denoise)
    → 03_final/*_Vocals_dry_dry.wav (262曲, 29.2h)
      → recut.py (無音検出で15-30sに分割 + silenceremoveで前後無音除去)
        → speaker1-plus/ (4507区間, 29.2h)
          + speaker1/ (1153区間, ~7.2h)
            = train_data/ (5603区間, ~36.4h 正味ボーカル)
```

### データセット統計

| ディレクトリ | ファイル数 | 時間 | 由来 |
|------|:--:|:--:|------|
| `train_data/speaker1/` | 1153 | ~7.2h | 初期 93 曲の分割 |
| `train_data/speaker1-plus/` | 4450 | ~29.2h | 追加 262 曲の分割 |
| **学習合計** | **5603** | **~36.4h** | — |
| 期限切れ/短すぎるため `ft_dataset.py` が自動スキップ | ~60 | — | max=30s, min=1s |

### 分割ロジック (recut.py)

```
入力: 03_final/*_Vocals_dry_dry.wav (ドライボーカルのみ取得、_other はスキップ)

分割方針:
  ≤30s → 曲全体を保持
  >30s → 15-30s の窓内で無音点を探して分割
    優先度: 長い停止≥2s > 自然な段落末≥22s > 文間の隙間≥15s > 29.5sで強制分割

各区間の切り出し後: ffmpeg silenceremove (-35dB, 0.3s) で前後無音を除去
最終チェック: <1s は削除、>31s は学習時 dataloader が自動スキップ
```

---

## 10. LR Schedule

### 60k ステップ Cosine Warmup

```
lr(t):
  0 ~ 3000歩:    linear warmup  0 → 1e-4
  3000 ~ 60000歩: cosine decay  1e-4 ↘ 1e-6
```

| ステップ | lr | フェーズ |
|:--:|:--:|------|
| 0 | 0 | 開始 |
| 3k | 1e-4 | warmup 終了 |
| 15k | 7.5e-5 | 主力学習 |
| 30k | 5e-5 | 中盤収束 |
| 45k | 2.5e-5 | 微調整 |
| 60k | 1e-6 | 終了 |

旧方式(ExponentialLR gamma=0.999996)との比較: 15000歩後も lr は初期値の94%で、ほぼ一定 lr。

可視化: `temp/temp_0502/lr_schedule_60k.html`

---

## 11. Speaker Embedding 固定化案

### 目的

推論時に target 参照音声へ依存せず、花丸の音色を完全にモデル重み内部の表現に持たせる。

### 実装 (train_yingmusic_ft_spkemb.py)

**学習時:**
```python
self.spk_embedding = nn.Embedding(1, 192)   # 花丸 = id 0
# ⚠️ avg_style (5603件平均) による初期化は無効と確認済み — cos=0.72 vs 単一参照, 勾配=0 で修正不能
# ✅ 単一の高品質参照 CAMPPlus (hanamaru_prompt_style.pt) に変更
style = self.spk_embedding(torch.zeros(B, dtype=torch.long, device=self.device))
# spk_embedding は backward に参加するが、勾配が非常に弱く、初期化品質がすべてを決める
# CFM + LengthRegulator + spk_embedding の3者を同時最適化
```

**推論時 (inference_spkemb.py):**
```python
style = spk_emb(torch.zeros(1, dtype=torch.long, device=device))
# mel2 + S_ori は花丸参照から一度だけ抽出し、永続キャッシュする
# 推論コマンドラインに --target パラメータは不要
```

### 推論コマンド (target 音声なし)

```powershell
cd YingMusic-SVC
python inference_spkemb.py `
    --source <任意音声> `
    --checkpoint output_models/yingmusic_spkemb_60k/ft_model.pth `
    --config configs/YingMusic-SVC.yml `
    --diffusion-steps 50
```

初回実行時は `花丸-平-voice.mp3` から mel2 と S_ori を抽出し、`output_models/ref_mel2.pt` + `output_models/ref_S_ori.pt` へキャッシュする。以後は即時ロード。

---

## 12. 学習パラメータ (version: yingmusic_spkemb_60k)

### 実行コマンド

```powershell
cd YingMusic-SVC
accelerate launch --config_file accelerate_config.yaml `
    train_yingmusic_ft_spkemb.py `
    --config configs/YingMusic-SVC.yml `
    --pretrained-ckpt YingMusic-SVC-full.pt `
    --dataset-dir train_data `
    --run-name yingmusic_spkemb_60k `
    --save-every 10000
```

### 学習設定一覧

| パラメータ | 値 |
|------|:--:|
| データセット | speaker1 + speaker1-plus = 5603件 / 36.4h |
| 開始点 | YingMusic-SVC-full.pt (CFM + LengthRegulator) |
| base_lr | 1e-4 |
| schedule | warmup 3k → cosine decay 60k |
| lr_min | 1e-6 |
| batch_size | 1 (12GB VRAM 制限) |
| 速度 | ~1.1 it/s |
| 総ステップ | 60000 (~7.2 epochs) |
| 予想時間 | ~15 時間 |
| 保存頻度 | 10000 歩ごと |
| Timbre Shifter | OpenVoice se_db.pt (100004人) |
| Speaker Embedding | nn.Embedding(1,192), avg_style初期化 |
| エネルギー均衡 loss | ✅ balance_loss=True |
| Style Residual | ✅ return_style_residual=True |

### 出力モデル

| ステップ | ファイル |
|:--:|------|
| 10000 | `DiT_epoch_*_step_10000.pth` |
| 20000 | `DiT_epoch_*_step_20000.pth` |
| 30000 | `DiT_epoch_*_step_30000.pth` |
| 40000 | `DiT_epoch_*_step_40000.pth` |
| 50000 | `DiT_epoch_*_step_50000.pth` |
| 60000 | `ft_model.pth` (最終) |

---

## 13. 主要ファイル一覧

> ⚠️ 以下のパスは開発環境の履歴（`temp/temp_0502/` = 開発作業ディレクトリ、`seed-vc/` = upstream project）。本リポジトリでは主要ファイルをルートと `modules/` 配下に整理済み。

### 学習スクリプト

| ファイル | リポジトリ内位置 | 役割 |
|------|------|------|
| `train_yingmusic_ft_spkemb.py` | ルート | **主学習スクリプト（spk_embedding 含む）** |
| `train_yingmusic_ft_cosine.py` | ルート | Cosine warmup 学習スクリプト |
| `train_yingmusic_ft.py` | ルート | 基本学習（lr=5e-6） |
| `_train_spkemb_v2.py` | ルート | v2 学習スクリプト（環境変数注入） |

### 推論スクリプト

| ファイル | リポジトリ内位置 | 役割 |
|------|------|------|
| `inference_spkemb.py` | ルート | **target なし推論（spk_embedding 固定化）** |
| `my_inference.py` | ルート | YingMusic 公式推論（target 必須） |
| `inference.py` | ルート | Seed-VC V1 推論 |

### データツール

| ファイル | リポジトリ内位置 | 役割 |
|------|------|------|
| `recut.py` | ルート | ドライボーカル分割 + silenceremove |
| `extract_avg_campplus.py` | ルート | 平均 CAMPPlus 抽出 |
| `clean_silence.py` | ルート | 無音処理 |

### スケジューラと設定

| ファイル | リポジトリ内位置 | 役割 |
|------|------|------|
| `optimizers_cosine.py` | ルート | CosineWarmupScheduler |
| `optimizers.py` | ルート | optimizer 構築 |
| `accelerate_config.yaml` | ルート | accelerate 設定 |

### YingMusic コアモジュール

| ファイル | リポジトリ内位置 |
|------|------|
| `length_regulator.py` | `modules/` (YingMusic版, 238行) |
| `flow_matching.py` | `modules/` (YingMusic版, 667行, エネルギー均衡 loss 含む) |
| `diffusion_transformer.py` | `modules/` (YingMusic版, 564行, style_r チャンネル含む) |
| `commons.py` | `modules/` (モデル構築 + checkpoint 読み込み) |
| `YingMusic-SVC.yml` | `configs/` |

### 学習成果物

| パス | 内容 |
|------|------|
| `output_models/yingmusic_spkemb_60k/` | spkemb 60k 学習出力 |
| `output_models/spkemb_v3_lr3e-5_20k/` | v3 20k 学習出力 |
| `output_models/hanamaru_prompt_style.pt` | 単一 CAMPPlus (192次元, v3初期化) |
| `output_models/ref_mel2.pt` | キャッシュ済み参照 mel (推論用) |
| `output_models/ref_S_ori.pt` | キャッシュ済み参照 semantic (推論用) |

### 2.1 全体像：データフロー

```
推論:
  source.wav → Whisper → S_alt → LengthRegulator → cond
  target.wav → Whisper → S_ori → LengthRegulator → prompt
    │                                CAMPPlus → style
    │                                RMVPE → F0_alt/F0_ori
    │                target_mel
    │                    │
    │    cond + prompt + style + mel + F0
    │                    │
    └────────────────────┼──→ CFM.inference → mel → BigVGAN → 出力.wav
```

学習時はさらに一段階ある。花丸音声をまず OpenVoice Timbre Shifter に通して音色を攪乱し、その後 Whisper で semantic token を抽出して S_alt（内容=花丸、音色=ランダム人物）を作る。これにより、CFM は「任意音色の内容 token」から花丸の mel を復元することを学ぶ。

### 2.2 主要推論パラメータ

| パラメータ | 役割 | 歌唱推奨 |
|------|------|:--:|
| `diffusion_steps` | Euler 積分ステップ数 | 20-50 |
| `f0_condition` | F0 を注入（歌唱では必須） | True |
| `inference_cfg_rate` | CFG 強度（スタイル追従） | 0.7 |
| `auto_f0_adjust` | 音高範囲の自動整合 | False |
| `semi_tone_shift` | 手動半音シフト | 0 |
| `length_adjust` | 生成ウィンドウのスケール | 1.0 |

### 2.3 各モジュール一覧

| モジュール | 入力 | 出力 | 役割 |
|------|------|------|------|
| Whisper-small | 16kHz 波形 (B, T_16k) | semantic token (B, T_sem, 768) | 音声内容を抽出（理論上は話者非依存） |
| RMVPE | 16kHz 波形 (B, T_16k) | F0 (B, T_f0) 各フレーム Hz | 旋律/基本周波数を抽出 |
| Length Regulator | semantic token + F0 + 長さ | cond (B, T_mel, 768) | 時間軸整合 + F0 注入 + VQ 量子化 |
| CAMPPlus | 16kHz 波形 → 80band fbank | style (B, 192) | グローバル音色エンコード |
| CFM / DiT | ノイズ + cond + style + prompt mel | 速度場予測 | 拡散モデル、段階的に mel を復元 |
| BigVGAN | mel (B, 128, T_mel) | 波形 @ 44100Hz | vocoder |
| OpenVoice ToneColorConverter | 音声 + source se + target se | 音色変換後音声 | 学習時の音色攪乱 |

---

## 3. V1 学習フローの段階別分解

**ファイル**：`seed-vc/train.py`、クラス `Trainer.train_one_step()`

### Step 1 — Timbre Shifter（音色攪乱）
```
花丸音声 → extract_se(花丸) → speaker_emb (花丸) を取得
se_db からランダムに他人の speaker_emb を1つ抽出
ToneColorConverter.convert(花丸音声, 花丸_se, 他人_se) → perturbed_wav
```
効果：花丸の「内容 + 他人の音色」。反復ごとにランダムな人物へ変える（100004人から抽出）。

### Step 2 — Whisper（semantic 抽出）
```
S_ori = whisper.encoder(waves_16k)          # 花丸の元 semantic
S_alt = whisper.encoder(perturbed_wav_16k)  # 攪乱後 semantic
```

### Step 3 — RMVPE（F0）
```
F0_ori = rmvpe(waves_16k)  # 花丸の旋律
```

### Step 4 — Length Regulator
```
cond_ori = length_reg(S_ori, target_len=mel_len, f0=F0_ori)
cond_alt = length_reg(S_alt, target_len=mel_len, f0=F0_ori)
```

### Step 5 — ランダム Prompt 長
```
prompt_len = randint(0, mel_len-1)
cond = cond_alt のコピー
cond[:prompt_len] = cond_ori[:prompt_len]   # prompt 区間では元 semantic を使う
```

### Step 6 — CAMPPlus（グローバルスタイル）
```
style = campplus(花丸の元音声)  # (B, 192)
```

### Step 7 — CFM 学習
```
loss = cfm(mel（ground truth）, mel_len, prompt_len, cond, style)
```
内部：ランダム拡散時刻 t → ノイズ z → 直線補間 x_t → DiT が速度場 v_pred を予測 → L1(v_pred, v_target)

### Step 8 — 逆伝播
```
loss_total = cfm_loss + commitment_loss*0.05 + codebook_loss*0.15
backward → clip_grad → optimizer.step
```

### 学習 vs 推論の比較

| | 学習 | 推論 |
|------|------|------|
| source semantic token | 花丸+ランダム音色攪乱 | 任意人物の Whisper 出力 |
| prompt semantic | 元の花丸（10% 確率で長さ0） | target 花丸参照音声 |
| mel | 花丸の実 mel（ground truth） | ノイズから段階的に復元 |
| style | 花丸の CAMPPlus | target 花丸の CAMPPlus |
| CFM の挙動 | 速度場を学習（L1 loss） | Euler 積分で mel を復元 |

---

## 4. Seed-VC V2 アーキテクチャと AR 失敗分析

### 4.1 V2 アーキテクチャ

```
V2 推論チェーン (--convert-style true):
source音声 → Whisper → HuBERT → BSQ(狭, codebook=32) → narrow token
target音声 → Whisper → HuBERT → BSQ(狭, codebook=32) → narrow token
                                                              ↓
narrow token → AR(自己回帰) → wide token (codebook=2048) → CFM → BigVGAN → 音声
                                                              ↑
target音声 → Whisper → HuBERT → BSQ(広, codebook=2048) ──→ prompt
                                 → CAMPPlus ──→ style
                                 → mel_fn ──→ prompt mel
```

### 4.2 AR モデル失敗の根本原因

**問題**：`--convert-style true`（AR 有効）時、出力が文字化け/叫び声のようになる。

**連鎖的な切り分け**：

1. **AR は花丸の narrow token しか見ない状態で 6000 歩微調整された**。学習データは花丸1人分の narrow token（Whisper→HuBERT→BSQ 量子化）のみ。
2. **推論時には natori_sana の narrow token が入力される**。AR はこの token 分布を見たことがない（人物ごとの phonetic/prosodic pattern に差がある）。
3. **AR は自己回帰 Transformer** であり、入力分布に敏感。narrow token には実践上、残留話者情報の漏れがある（Whisper+HuBERT の disentanglement は完全ではない）。
4. **1ステップの予測誤りが累積** → wide token 系列全体が崩壊 → CFM がそれに従って mel を生成 → 叫び声化。

### 4.3 試したデバッグ（すべて失敗）

| 試行 | 結果 |
|------|------|
| temperature=0.3→0.1, top_p=0.5→0.3 | 依然として叫び声（問題は入力分布で、サンプリングのランダム性ではない） |
| AR 初期 checkpoint (step=500) | 依然として叫び声（最初から花丸 narrow token に過学習していた） |
| 微調整 CFM + HF デフォルト AR（混在） | CUDA crash（vocab size 不一致） |

### 4.4 結論

**AR 微調整は、同一話者の異なるスタイル変換にのみ適用可能**（例：花丸の通常話声→花丸の別感情）。話者横断シナリオ（natori→花丸）では、微調整後の AR は使えない。

より根本的には、このアーキテクチャは**「任意話者の発話内容→別人物の話し方」への移植を実現できない**。AR は token レベルのスタイル着色しか行わず、semantic content を変えられない。花丸の口癖/リズム/息継ぎパターンは semantic と結びついており、narrow→wide token の写像だけでは natori の narrow token からそれらの特徴を復元できない。

---

## 5. V2 推論における target 参照音声の役割

AR が崩れても、target は V2 内でまだ3箇所に効いている：

| target の有効箇所 | 形式 | 次元 | 役割 |
|------|------|:--:|------|
| AR narrow token コンテキスト | `tgt_narrow_reduced` を AR 入力 prefix に連結 | (1, T, —) | AR に生成対象のスタイルを伝える |
| AR wide token テンプレート | `target_content_indices` を `prompt_target` として使用 | (1, T, —) | AR の出力形式の参照 |
| CFM prompt mel | `target_mel` が拡散前 T フレームを固定 | (1, 80, T) | 拡散の初期条件 |
| CFM style | CAMPPlus 192次元 | (1, 192) | 各 DiT 層の AdaLN 変調 |
| CFM prompt_condition | LengthRegulator 出力 | (1, T, 512) | 条件系列 prefix |

---

## 6. YingMusic-SVC vs V1 モジュール別比較

### 6.1 Length Regulator

| | V1 | YingMusic |
|------|:---:|:---:|
| コード量 | 141 行 | 238 行 |
| 主な追加 | — | `use_style_residual` MLP |

YingMusic には2つのサブネットワークが追加されている：
```python
self.f0_to_style_proj = nn.Linear(768, 192)    # F0 を style 空間へ射影
self.f02style_mlp = Sequential(                 # フレーム単位の style residual 生成器
    Linear(384→192), Mish, Linear(192→192)
)
```

forward 時：F0_emb を 192 次元へ射影し、グローバル style と連結 → MLP → tanh → residual (B, T_mel, 192)。振幅は `alpha * RMS(グローバルstyle)` に制限され、有声フレームでのみ有効。

### 6.2 Flow Matching（CFM）

| | V1 | YingMusic |
|------|:---:|:---:|
| コード量 | 167 行 | 667 行 |
| 追加 | — | `style_r` チャンネル + エネルギー均衡 loss |

**差分 A**：`style_r` パラメータが LengthRegulator から DiT estimator へそのまま渡される。

**差分 B**：エネルギー均衡損失 (`balance_loss=True`)。mel の 128 周波数帯について、内容エネルギー + ノイズ水準に基づく適応重み `w_bc` を生成し、高域の弱エネルギー区間を重くする。学習目標は均一 L1 ではなく、`L1(pred*sqrt(w), target*sqrt(w))`。

### 6.3 DiT (Diffusion Transformer)

| | V1 | YingMusic |
|------|:---:|:---:|
| コード量 | 537 行 | 564 行 |
| 差分 | — | `style_r` チャンネルが追加 |

### 6.4 独自補助モジュール

| モジュール | パス | 役割 |
|------|------|------|
| `f0_normalization.py` | YingMusic 専用 | WORLD vocoder 抽出 → F0 正規化 → 合成 |
| `f0_fix.py` | YingMusic 専用 | F0 攪乱：jitter/glide/jump |

### 6.5 Timbre Shifter

| | V1 | YingMusic |
|------|------|------|
| モデル | OpenVoice ToneColorConverter | RVC-based（120人で学習） |
| 重み | `se_db.pt` (100004人) | 未公開 |
| ローカル | ✅ | ❌ |

### 6.6 まとめ：YingMusic が V1 より多く持つもの

```
V1 基盤:
  ├── CFM (DiT + 標準 flow matching L1)
  ├── Length Regulator (Wavenet + VQ + F0 embed)
  └── Timbre Shifter (OpenVoice, 100004人)

YingMusic の V1 への追加:
  ├── Length Regulator に追加 ──→ use_style_residual MLP (F0-aware timbre adaptor)
  ├── CFM に追加 ──→ style_r チャンネル (フレーム単位 style residual 注入)
  │              ──→ energy balance loss (高域加重、mel 周波数の均衡)
  ├── Timbre Shifter を置換 ──→ RVC-based (120歌手、未公開)
  └── 補助ツール ──→ f0_fix.py, f0_normalization.py
```

---

## 7. RVC Timbre Shifter の現状と代替案

### 7.1 確認済みの事実

- YingMusic の RVC timbre shifter は学習時専用（音色攪乱）であり、**推論 checkpoint 内には含まれない**
- `YingMusic-SVC-full.pt` には `cfm` + `length_regulator` のみが含まれ、shifter 重みはない
- 公式は RVC shifter の事前学習重みを公開していない
- この shifter は 120 人の singer で学習されており、歌唱シナリオ向けに最適化されている

### 7.2 代替案比較

| 案 | 話者横断 | 花丸類似度 | 作業量 |
|------|:---:|:---:|:--:|
| OpenVoice で RVC を代替 | ✅ | やや弱い | ✅ 実装済み（本リポジトリルート） |
| shifter をスキップ | ❌ | 強い | 小 |
| RVC shifter を自前学習 | ✅ | ✅ | 巨大 |

### 7.3 結論：OpenVoice で代替済み

- `se_db.pt`（100004 人 embedding）は OpenVoice からダウンロードが必要（リポジトリには含まれない）
- 自作学習コードでは OpenVoice ToneColorConverter + se_db ランダムサンプリングにより RVC shifter を代替済み
- 100004 人のカバレッジ >> 120 人であり、話者横断の汎化が良い
- 代償として singing-specific 品質はやや低下（高域/ビブラート保持は RVC 専用学習版に劣る）
- Timbre shifter はモデル内部にないため、置換しても checkpoint 読み込みに影響しない
- 学習コードパス：`train_yingmusic_ft_spkemb.py`（リポジトリルート）

---

## 14. spkemb 60k 失敗の根本原因（2026-05-04）

### 14.1 重要な数値

```
hanamaru_avg_style.pt (5603件平均) vs 花丸-平-voice.mp3 CAMPPlus:
  cos = 0.724498     (偏差 28%)
  L2  = 8.4301       (avg norm=8.6 vs single norm=12.2)

spkemb 60k spk_embedding vs avg_style:
  cos = 1.000000     (学習の最初から最後まで動かなかった)
  L2  = 0.0000
```

### 14.2 重み偏移の比較

| モデル | lr | ステップ | CFM cos | 音色 |
|------|:--:|:--:|:--:|------|
| V1 15000歩 | 5e-6 | 15000 | 0.9996 | ✅ |
| exp1 | 5e-6 | 12000 | 0.9997 | ✅ |
| cosine | 1e-4 | 3000 | 0.999999 | ✅ (停止が早すぎる) |
| spkemb 40k | 1e-4 | 40000 | 0.921 | ❌ 重すぎる |
| spkemb 60k | 1e-4 | 60000 | ~0.90 | ❌ フェイダン状態 |

### 14.3 3つの原因の重なり

1. **初期化ミス**：avg_style が MSST 環境ノイズで汚染された
2. **spk_embedding の勾配がゼロ**：学習で初期化を修正できない
3. **CFM が誤った style の下で過学習**：誤った方向へ 60k 歩進んだ

### 14.4 検証実験

spkemb 60k CFM + `my_inference.py`（リアルタイム CAMPPlus）で推論 → 音色は「まあまあ」回復するが、声が「厚い」（エネルギー均衡 loss の副作用）。証明されたこと：
- 音色差 → avg_style が原因
- 厚み → CFM 過学習が原因

### 14.5 mel2 実験

| mel2 ソース | 効果 |
|-----------|------|
| 花丸-平-voice.mp3 | ✅ 正常 |
| 283件学習データ平均 | ❌ かなり悪い |
| 全ゼロベクトル | ❌ かなり悪い |

mel2 は単純平均できない（時間方向のダイナミクスを消してしまう）し、ゼロにもできない（拡散のアンカーを失う）。単一の高品質参照が最適解。

---

## 15. v3 実験案（2026-05-04 ✅ 完了）

### 目標
単一 CAMPPlus で spk_embedding を初期化し、低 lr で CFM 偏移を制御し、音色が良く自然なモデルを得る。

### パラメータ

| パラメータ | 値 |
|------|:--:|
| データセット | 5603件 / 36.4h |
| 開始点 | YingMusic-SVC-full.pt |
| init | **hanamaru_prompt_style.pt** (単一 CAMPPlus) |
| base_lr | 3e-5 |
| schedule | warmup 3k → cosine 20k |
| lr_min | 1e-6 |
| 総ステップ | 20000 |
| 実時間 | ~5 時間 |

### 重み偏移（学習全体）

| ステップ | CFM cos | LR cos | spk cos | フェーズ |
|:--:|:--:|:--:|:--:|------|
| 5000 | 0.998109 | 0.961228 | 1.000000 | 初期は無音化あり |
| 10000 | 0.996313 | 0.960366 | 1.000000 | 無音化が大きく減少 |
| 15000 | 0.995953 | 0.959579 | 1.000000 | ほぼ収束 |
| 20000 | 0.995964 | 0.959710 | 1.000000 | 15k→20k はほぼ変化なし |

### spkemb 60k との比較

| | spkemb 60k | v3 |
|------|:--:|:--:|
| spk init | avg_style ❌ | prompt_style ✅ |
| base_lr | 1e-4 | **3e-5** |
| 総ステップ | 60000 | **20000** |
| CFM cos 終値 | ~0.90 | **0.996** |
| 推論方式 | spkemb/CAMPPlusリアルタイム | **CAMPPlusリアルタイム (最良)** |
| 音色 | 厚い | **最も正確** |

### 推論結論

- **v3 20k + my_inference.py (CAMPPlusリアルタイム)** = 最良推論案、音色が最も正確
- **v3 20k + inference_spkemb.py (spk_embeddingルックアップ)** = リアルタイム CAMPPlus より効果が劣る（spk cos=1.000 でも）
- 初期 5k checkpoint には明確な無音化がある（style_r 未収束）、15k では完全に解消

### 出力

| ステップ | ファイル |
|:--:|------|
| 5000 | `spkemb_v3_lr3e-5_20k/DiT_epoch_*_step_05000.pth` |
| 10000 | `spkemb_v3_lr3e-5_20k/DiT_epoch_*_step_10000.pth` |
| 15000 | `spkemb_v3_lr3e-5_20k/DiT_epoch_*_step_15000.pth` |
| 20000 | `spkemb_v3_lr3e-5_20k/ft_model.pth` |

### 追加ファイル

| ファイル | 役割 |
|------|------|
| `_train_spkemb_v2.py` | lr/init の環境変数注入に対応した学習スクリプト |
| `run_spkemb_v2_experiments.py` | 一括実験ランチャー |
| `hanamaru_prompt_style.pt` | 単一 CAMPPlus (norm=12.2) |
| `batch_matrix_infer.py` | マトリクス推論スクリプト（進捗バー付き） |
| `inference_spkemb_zero_mel2.py` | ゼロ mel2 比較実験 |
| `inference_spkemb_avg_mel2.py` | 平均 mel2 比較実験 |

---

## 16. 4モデルマトリクス比較の最終結論（2026-05-04）

### 16.1 実験設計

8曲 × 5モデル（cosine/exp1/spkemb 60k/v3 20k campplus + V1 15000参照）、出力ファイルは `矩阵对比/output/` を参照。

### 16.2 中核的な発見

**清涼 ←→ 厚い は連続スペクトラムである：**

```
cosine(3k)  exp1(12k)  v3(20k)  spkemb(60k)
  最も清涼    清涼で柔らかい  中間      最も厚い
  CFM不変     CFM不変       CFM微変化  CFM大幅変更
  cos=1.0    cos=0.9997 cos=0.996 cos=0.90
```

- **cosine/exp1 が清涼** = CFM がほぼ動かない + lr 低/ステップ少で、事前学習モデルのクリーンな質感を保持
- **v3 は音色が最も正確** = 正しい CAMPPlus init + 適度な学習、CFM cos=0.996 は安全圏
- **60k は厚い** = エネルギー均衡 loss が CFM mel 生成経路を過度に学習させたが、CAMPPlus リアルタイム推論で音色方向は救えた

### 16.3 実戦での選定

| シーン | 推奨 | 推論パラメータ |
|------|------|------|
| 最も花丸に近い | V1 15000歩 | diff=20, f0=True |
| YingMusic で音色が最も正確 | v3 20k campplus | diff=100, f0=True |
| YingMusic で清涼感優先 | cosine 3000 | diff=50, f0=True |
| YingMusic バランス型 | exp1 12000 | diff=50, f0=True |
| 厚いスタイル | spkemb 60k | diff=50, f0=True |

**すべての YingMusic モデル**は `my_inference.py` + `--target 花丸-平-voice.mp3`（CAMPPlus リアルタイム）で統一する。`inference_spkemb.py`（target なし）はリアルタイム方式より効果が劣る。

### 16.4 学習方法論まとめ

| 法則 | 内容 |
|------|------|
| **初期化の法則** | spk_embedding 勾配≈0、初期化品質 = 最終品質 |
| **CFM 安全圏** | cos > 0.95：清涼；cos ~0.90：厚いがまだ使用可能（CAMPPlusリアルタイムで救える） |
| **清涼 vs 正確** | 両立しにくい。CFM が動かなければ清涼だが音色が不正確な場合があり、CFM を動かすと正確だが厚くなる |
| **データ量のパラドックス** | 1153件で最も清涼な効果を出せる（cosine/exp1）；5603件+大lr は逆に厚くなる |
| **最良推論** | CAMPPlus リアルタイム > spk_embedding ルックアップ（cos=1.000 でも） |

---

## 8. YingMusic 微調整の実行ルート

### 8.1 目的

YingMusic-SVC-full.pt（CFM + LengthRegulator、style_residual 含む）を花丸データで微調整し、話者横断能力を持つ専用モデルを得る。

### 8.2 完了済み事項

1. ✅ **学習スクリプト** `train_yingmusic_ft_spkemb.py`、V1 の `train.py` をベースに改修
2. ✅ **config パス変更** → `YingMusic-SVC.yml`（`use_style_residual` パラメータ）
3. ✅ **モデル構築変更** → YingMusic の `build_model()` が style_residual 付き LengthRegulator を正しく構築
4. ✅ **YingMusic checkpoint 読み込み** → `YingMusic-SVC-full.pt` の `cfm` + `length_regulator` 重み
5. ✅ **OpenVoice Timbre Shifter** → V1 `train.py` から ToneColorConverter + se_db ランダムサンプリングを移植
6. ✅ **エネルギー均衡 loss 有効化** → `balance_loss=True`（YingMusic CFM に実装済み）
7. ✅ **複数回の学習完了** → spkemb 60k (フェイダン状態→保存), v3 20k (✅主力), cosine 3000 (✅最清涼), exp1 12000 (✅清涼で柔らかい)

### 8.3 期待される効果

- CFM は YingMusic 事前学習 + エネルギー均衡 loss → 歌唱表現力向上
- Length Regulator は style_residual MLP → 動的な音色調整
- OpenVoice shifter により話者横断の汎化を維持
- 花丸データでの微調整 + 学習可能 spk_embedding → 音色が花丸へ近づく

### 8.4 関連ファイル一覧

| ファイル | リポジトリ内位置 |
|------|------|
| YingMusic 推論スクリプト | `my_inference.py`（ルート） |
| YingMusic モデル設定 | `configs/YingMusic-SVC.yml` |
| YingMusic checkpoint | `YingMusic-SVC-full.pt` (HF: GiantAILab) |
| YingMusic length_regulator | `modules/length_regulator.py` |
| YingMusic flow_matching | `modules/flow_matching.py` |
| YingMusic DiT | `modules/diffusion_transformer.py` |
| OpenVoice se_db | `modules/openvoice/checkpoints_v2/converter/se_db.pt` |
| **自作学習スクリプト** | **`train_yingmusic_ft_spkemb.py`（ルート）** |
| **自作推論スクリプト** | **`inference_spkemb.py`（ルート）** |

---

## 9. 重み偏移分析（V1 vs YingMusic）

### 9.1 分析方法

`compare_weights.py` / `batch_compare.py` / `compare_v1.py` を使い、微調整 checkpoint と各事前学習ベースをパラメータ単位で cosine similarity と L2 距離により比較する。

### 9.2 V1 重み偏移（ベース：Seed-VC DiT 事前学習）

| 実験 | ステップ | CFM cos | LR cos | 特徴 |
|------|:--:|:--:|:--:|------|
| my_run (R1) | 1750 | 0.9998 | 0.916 | 48件、LR が一度に整合 |
| hanamaru_full_fav (R2) | 6000 | **0.9996** | **0.916** | 1153件、CFM はほぼ動かない |
| hanamaru_full_fav_step15000 | 15000 | **0.9996** | **0.916** | ★ベスト、継続学習しても追加変化なし |

**結論**：V1 の CFM は学習全体を通して cosine で 0.0004 しか偏移しておらず、ほぼ事前学習のまま。LR は最初の 500 歩で整合のため一度調整された後、cos=0.916 で固定される。これは「CFM を軽く触る + F0 誘導」戦略が極めて効率的であることを示す。

### 9.3 YingMusic 重み偏移（ベース：YingMusic-SVC-full.pt）

| ステップ | CFM cos | CFM ΔL2 | LR cos | LR ΔL2 |
|:--:|:--:|:--:|:--:|:--:|
| 5000 | 0.980 | 497 | 0.942 | 22 |
| 10000 | 0.958 | 763 | 0.924 | 37 |
| 20000 | 0.934 | 1017 | 0.896 | 53 |
| 30000 | 0.924 | 1126 | 0.881 | 60 |
| **40000** | **0.921** | **1155** | **0.876** | **62** |
| 予測 60000 | ~0.918 | — | ~0.873 | — |

### 9.4 比較解釈

| | V1 | YingMusic |
|------|------|------|
| CFM 偏移量 | 0.0004 (ほぼ不変) | 0.079 (大きく書き換え) |
| LR 偏移量 | 0.084 (一度だけジャンプ) | 0.124 (継続学習) |
| 学習モード | 軽い微調整 | 深い適応 |
| 強み | 旋律安定性 ⭐⭐⭐⭐⭐ | 自然さが高い（エネルギー均衡 + style_r） |
| 代償 | 高域が刺さる | CFM の書き換え幅が大きく、収束により多くのステップが必要 |

V1 は LR だけを動かし CFM を動かさなくても十分に良いことを証明した。YingMusic は追加でエネルギー均衡 loss とフレーム単位 style_residual を導入しているため、CFM も合わせて書き換わらないと性能を発揮しない。試聴では YingMusic の自然さが V1 より優れていることを確認済み。
