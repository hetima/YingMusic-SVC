> ⚠️ **最新更新（2026-05-04）**：YingMusic 4モデル（60k spkemb / exp1 12000 / cosine 3000 / v3 20k campplus）について、8曲でマトリクス比較を完了。**それぞれ長所と短所がある**：v3 20k campplus は音色が最も正確、cosine と exp1 はより清涼、60k spkemb はフェイダン状態ながら独特の厚みがある。
>
> ⚠️ **履歴更新（2026-05）**：原作者（GiantAILab）は現在も**学習コードをオープンソース化していない**。公式リポジトリは今も inference-only。ただし、こちらではすでに**自前の学習スクリプトを作成済み**で、3回の反復を経ている。

**YingMusic-SVC** の**コミュニティ案や具体的なチュートリアル**は現時点では比較的少ない（2025年末公開のプロジェクト）。公式は **inference-only** が中心で、**学習/微調整コードはまだ公開されていない**（公式 GitHub リポジトリには推論スクリプト、accompany separation、Gradio app はあるが、`train.py` や完全な学習 pipeline はない）。

### 1. 公式リポジトリとコミュニティの現状
- **公式リポジトリ**：https://github.com/GiantAILab/YingMusic-SVC （MIT License、Seed-VC をベースに拡張）。
  - 含まれるもの：`accom_separation/`（伴奏分離）、`gradio_app.py`（GUI）、`my_inference.py` / `my_infer.sh`（CLI 推論）、`configs/`、事前学習 checkpoint（HF またはスクリプト経由でダウンロード）。
  - **学習コードなし**：README は inference 関連のみ。論文では3段階学習（CPT → Robust SFT → Flow-GRPO RL）が説明されているが、直接実行できるスクリプトは提供されていない。
  - Forks：少数（例：juanjosehr14/YingMusic-SVC）。主にミラーまたは軽いラッパーで、追加の学習対応はない。
  - Discussions/Issues：GitHub と HF ページ上の議論は少ない（stars ~130、forks ~13）。現時点で活発なコミュニティ学習共有や Discord サーバーはない。
  - HF ページ：https://huggingface.co/GiantAILab/YingMusic-SVC （model card は短く、主に GitHub への誘導で、追加の学習例はない）。

- **関連プロジェクト**：
  - **YingMusic-Singer**（同チーム）：ゼロショット歌声合成（SVS）。推論コードはあるが SVC ではない。
  - **Seed-VC**：YingMusic-SVC の基盤。微調整コード（`train.py`）と OpenVoice モジュールは、自作学習スクリプトの重要な依存。
  - コミュニティ議論は Reddit (r/so_vits_svc)、中国語フォーラム、論文評（例：themoonlight.io）などに散見されるが、多くは inference 利用や論文解釈であり、成熟した YingMusic-SVC 微調整コミュニティ案はない。

- **論文参考**：arXiv:2512.04793。論文は3段階学習（CPT → Robust SFT → Flow-GRPO RL）を説明しているが、論文中の RVC timbre shifter には追加学習データ（120人の歌手）が必要で、公式はこのモジュールの重みを提供していない点に注意。**自作学習コードでは、これを Seed-VC の OpenVoice se_db 攪乱方式へ置き換えている**（下記第3節）。

**結論**：原作者は確かに学習コードを公開していない。ただし、こちらでは Seed-VC アーキテクチャ + YingMusic の CFM/length_regulator モジュールをベースに、ゼロから利用可能な学習スクリプトを書き上げている。以下の第3節に実装済み案を記録する。

### 2. 具体的な使い方（Inference 部分、導入済みでそのまま利用可能）
使用中の venv（py310 + cu128）は準備済み。`E:\AIscene\AISVCs\YingMusic-SVC` に入り、venv を有効化した後：

#### **モード**
- **伴奏分離（Accompany Separation）**：実曲（ハモリ/伴奏あり）を処理し、きれいな lead vocal を出力する（後段 SVC の品質向上）。Band RoFormer ベース。
- **Zero-shot SVC**：source 歌唱音声（伴奏あり、または分離済み） → target 歌手参照音声 → melody + lyrics + target timbre を保持した変換歌声を出力。実世界の頑健性（harmony 干渉、F0 誤差）に強い。
- **全 pipeline**：分離 → SVC →（任意）伴奏とのミックス + BigVGAN vocoder。
- **Gradio GUI**：視覚的に操作でき、source/target のアップロードとパラメータ調整に対応。

#### **入力要件**
- **Source**：歌唱音声（wav/flac/mp3、44.1kHz 推奨、長さは 5-60s が最適。伴奏ありでもクリーン vocal でも可）。
- **Target Reference**：目標歌手の短い音声（1-30s、クリーンな話声/歌声どちらでも可。timbre 類似度に影響）。
- **任意**：semi-tone shift（性別をまたぐ場合は +12 がよく使われる）、伴奏ミックス重み γ。

#### **調整可能パラメータ**（主に Seed-VC の継承/拡張 + singing-specific）
- Diffusion steps / sampling steps（推論品質 vs 速度。10-50 推奨。12GB VRAM では OOM 防止のため低め推奨）。
- CFG rate / inference cfg（類似度 vs 多様性を制御）。
- F0 condition / pitch shift（半音移調）。
- Timbre shifter strength（RVC-based、disentanglement に影響）。
- Energy-balanced loss 関連（推論時に直接調整しないが、高域ディテールへ影響）。
- GUI では、ファイルアップロード後に steps、cfg、shift などの sliders が通常表示される（詳細は `gradio_app.py` のソースを参照）。

#### **手順**
1. **伴奏分離**（先に実行推奨。頑健性が上がる）：
   ```cmd
   cd YingMusic-SVC/accom_separation
   bash infer.sh   # または Windows 相当：python infer.py --input your_song.wav --output separated/
   ```
   - Band RoFormer モデルをダウンロード/使用し、lead vocal を出力する。

2. **CLI SVC 推論**：
   ```cmd
   cd YingMusic-SVC
   bash my_infer.sh
   ```
   - `my_infer.sh` または `my_inference.py` を編集し、パスやパラメータ（source、target、output、steps、cfg など）を調整する。
   - サンプルコマンド（ソース参照）：
     ```cmd
     python my_inference.py \
       --source separated_lead.wav \
       --target reference_singer.wav \
       --output outputs/result.wav \
       --diffusion_steps 25 \
       --cfg_rate 0.7 \
       --semi_tone_shift 0
     ```

3. **Gradio GUI**（最も手軽）：
   ```cmd
   python gradio_app.py
   ```
   - ブラウザで http://127.0.0.1:7860 にアクセス。
   - 操作：source（伴奏あり、または分離済み）と target reference をアップロードし、分離の有無や sliders を調整して生成する。

**VRAM 最適化（12GB 環境向け）**：
- fp16 / torch.compile を使う（対応していれば）。
- 短い音声 + 低 diffusion steps（real-time-ish なら 10-20）。
- `nvidia-smi` を監視する。OOM の場合は batch（推論は通常単一サンプル）または steps を下げる。
- BigVGAN vocoder は VRAM を使う場合があるため、必要なら段階的に実行する。

**出力**：変換後の歌唱音声（.wav）。melody/lyrics をよく保持し、timbre は target に近づく。特に noisy/harmony シーンに強い。

参考：公式 README、`gradio_app.py` / `my_inference.py` ソース、arXiv 論文の inference pipeline 部分。

### 3. 自作学習案（実装済み、原作者は未公開）

> **原作者（GiantAILab）は現在も学習コードを公開していない**。以下の学習スクリプトはすべてこちらで作成したもので、本リポジトリのルートに置かれている。

#### 学習スクリプト一覧

| ファイル | 用途 | 主要パラメータ |
|------|------|----------|
| `train_yingmusic_ft.py` | 基本版 | lr=5e-6、CAMPPlus リアルタイム抽出 speaker embedding |
| `train_yingmusic_ft_cosine.py` | Cosine 版 | lr=1e-4 warmup → cosine decay |
| `train_yingmusic_ft_spkemb.py` | **SpkEmb 版（現在の主力）** | 学習可能 `spk_embedding`（192次元）、warmup=3000, lr=1e-4→1e-6 |

#### コアアーキテクチャ変更：RVC Timbre Shifter → OpenVoice se_db 攪乱

論文中の RVC timbre shifter モジュールは、120人の歌手で追加事前学習する必要があり、公式はその重みを提供していない。こちらの代替案：

```
原文：音声 → RVC Timbre Shifter（利用不可）→ perturbed semantic → CFM
こちら：音声 → OpenVoice ToneColorConverter + se_db（100004人）→ perturbed semantic → CFM
```

具体的な流れ（`train_one_step` が中心）：

```
1. speaker embedding (se_batch) を抽出 ← OpenVoice ToneColorConverter
2. se_db から他人の embedding (ref_se) をランダムサンプリング   ← RVC 攪乱の代替
3. convert(waves, se_batch → ref_se)               ← 音色攪乱版音声を生成
4. Whisper/XLSR で semantic 抽出：S_ori（元音声）+ S_alt（攪乱後）
5. RMVPE で F0 を抽出
6. CAMPPlus（または spk_embedding）で style vector y を抽出
7. Length Regulator: S_alt + y → alt_cond + style_r
8. CFM Flow Matching: ランダム prompt_len で ori/alt cond を混合、balance_loss=True
9. cfm + length_regulator + spk_embedding を更新
```

**論文方式との主な差分**：

| モジュール | 論文方式 | こちらの実装 |
|------|----------|-----------|
| Timbre Shifter | RVC-based（120歌手で事前学習） | OpenVoice se_db ランダムサンプリング（100004人） |
| Speaker Embedding | CAMPPlus（リアルタイム） | 基本版は CAMPPlus、spkemb 版は学習可能 Embedding |
| F0-aware Adaptor | 論文特有 | `length_regulator(style=y, return_style_residual=True)` の style_r で間接実装 |
| Energy-balanced Loss | λ=0.4 | `balance_loss=True`（YingMusic CFM がネイティブ対応） |
| Harmony Augmentation | 伴奏を混入 | 未実装（前処理分離品質に依存） |
| Flow-GRPO RL | 3段階 RL | 未実装（12GB VRAM 不足 + reward model が複雑） |

#### 学習起動

```powershell
cd YingMusic-SVC

# SpkEmb 版（現在使用）
accelerate launch --config_file accelerate_config.yaml `
    train_yingmusic_ft_spkemb.py `
    --config configs/YingMusic-SVC.yml `
    --pretrained-ckpt YingMusic-SVC-full.pt `
    --dataset-dir train_data `
    --run-name yingmusic_spkemb_60k `
    --batch-size 1 `
    --max-steps 60000 `
    --save-every 10000
```

出力は `output_models/<run_name>/` に保存される。checkpoint の命名形式は `DiT_epoch_XXXXX_step_XXXXX.pth`、最終モデルは `ft_model.pth`。

#### 学習履歴

| 実験 | スクリプト | ステップ | lr | データ量 | 結果 |
|------|------|:--:|:--:|------|------|
| yingmusic_exp1 | ft.py | 12000 | 5e-6 | 1153件 | ✅ **YingMusic 主力**（CAMPPlusリアルタイム, 最もバランス型） |
| yingmusic_cosine | ft_cosine.py | 3000 | 1e-4 warmup | 1153件 | ✅ 単体試聴では改善あり |
| **yingmusic_spkemb_60k** | **ft_spkemb.py** | **60000** | **1e-4 cosine** | **5603件 / 36.4h** | **❌ フェイダン状態→保存（独特の厚みあり）** |
| **spkemb_v3_lr3e-5_20k** | **_train_spkemb_v2.py** | **20000** | **3e-5 cosine** | **5603件 / 36.4h** | **✅ v3 主力 (warmup 3k)** |

#### 学習中の重要な発見（2026-05）

**ft_dataset の長音声切り捨て修正**：元コードは >30s の音声を直接スキップしており、大量のデータを浪費していた。先頭 30 秒を切り出すよう修正し、長すぎる音声を捨てないようにした。

**スケジューラの断点再開 Bug 修正**：`load_checkpoint(load_only_params=True)` が再開時に optimizer と scheduler の状態を捨てており、停電後の再起動で lr が 0 から再 warmup していた。修正後は、初回の事前学習重み読み込み時は `load_only=True`、自分の checkpoint から再開するときは `load_only=False`。

**Loss 曲線の解釈**：
| 区間 | Loss | 原因 |
|------|:--:|------|
| 0~500 歩 | 0.75→0.58 急速に低下 | lr が 0 から上がり、モデルが素早く適応 |
| 500~3000 歩 | 0.58→0.72 上昇 | warmup が lr を 1e-4 まで押し上げ続け、勾配ノイズが拡大 |
| 3000~43000 歩 | 0.72→0.48 継続低下 | cosine decay が効き、収束が本筋に入る |
| 予測 60000 歩 | ~0.40~0.45 | 収束終盤に近い |

#### 22個のモデル重み偏移の全量分析

`batch_compare.py` を使い、全 22 checkpoint と `YingMusic-SVC-full.pt` の偏移を比較した：

| 実験 | ステップ範囲 | CFM cos 終値 | LR cos 終値 | 結論 |
|------|------|:--:|:--:|------|
| yingmusic_exp1 | 500→12000 | **0.9997** | 0.964 | ❌ lr=5e-6 が低すぎ、重みはほぼ変化なし |
| yingmusic_cosine | 1000→3000 | **0.999999** | 0.965 | ❌ 停止が早すぎ、warmup が終わった直後 |
| **yingmusic_spkemb_60k** | 5000→40000 | **0.921** | **0.876** | ✅ 継続学習、収束傾向は明確 |

spkemb の収束曲線：
```
05000  .980  ████████████████████████████████████████████
10000  .958  ████████████████████████████████████████
15000  .944  ████████████████████████████████████
20000  .934  ██████████████████████████████
25000  .927  █████████████████████████
30000  .924  ███████████████████
35000  .922  ██████████████
40000  .921  ██████████  ← cos 低下速度は 0 に近づき、収束中
```

#### Seed-VC V1 との重み偏移比較

| モデル | CFM cos | LR cos | 特徴 |
|------|:--:|:--:|------|
| V1 (★ベスト歌唱モデル) | **0.9996** | 0.916 | CFM はほぼ動かず、LR は一度だけ整合後に固定 |
| YingMusic 40000歩 | **0.921** | 0.876 | CFM が大幅に書き換わり、LR は style_residual を継続学習 |

V1 は「CFM を軽く触る + F0 誘導」だけでベスト歌唱効果を出せることを証明した。YingMusic は 20 倍のパラメータ量を追加で動かしているが、これはエネルギー均衡 loss と style_residual 機構に必要なもの。試聴では YingMusic の自然さが V1 より優れていることを確認済み（柔らかさ、高域ディテールが良い）。

#### F0→style 側枝の凍結分析

`f0_to_style_proj` と `f02style_mlp` の計 6 パラメータは 40000 歩後も cos=1.000 —— 完全に学習していない。根本原因は、gradient が `tanh → RMS normalize → clamp(max=1.0)` の3重切り詰めを通った後、ほぼゼロになること。効果には影響しない：F0 情報は `f0_embedding → x = x + f0_emb` の主経路から十分に CFM へ入っている。

### 4. 推奨される次の一手

- **v3 20k campplus**（lr=3e-5, 100 diff steps）→ **音色が最も正確**、YingMusic 歌唱の主力。
- **cosine 3000** → 声が**清涼**（8曲比較では exp1 と並んで最も清涼）。lr=1e-4 だがステップ数が短く、CFM cos=0.999999。
- **exp1 12000**（lr=5e-6）→ 音色が**清涼で柔らかい**。CFM cos=0.9997 で、cosine と同じ清涼系。
- **spkemb 60k**（lr=1e-4, CAMPPlusリアルタイム推論）→ 音色が**厚い**。CFM cos≈0.90。厚みが必要な曲向き。
- 代替案：Seed-VC V1 15000 歩を安定 baseline として使う（音色は最も似るが、やや刺さる）。
- 追加探索：cosine モデルを lr=1e-5 で微調整し、「清涼感」の限界を探る。


---

## 5. spkemb 60k 失敗原因分析（2026-05-04）

### 5.1 問題の現れ方

spkemb 60k のすべての checkpoint（5k/10k/15k/.../60k）で、出力音色が花丸から大きく逸脱した。最初期 checkpoint（5k歩）の効果が最も悪い。

### 5.2 根本原因

**`hanamaru_avg_style.pt` が実際の花丸音色から大きく逸脱していた。**

```
hanamaru_avg_style.pt  ← 5603件の MSST 分離成果物の CAMPPlus 単純平均
花丸-平-voice.mp3 CAMPPlus  ← 単一のクリーンで高品質な参照

cos = 0.724498     (方向が 28% ずれている)
L2  = 8.4301       (avg norm=8.6 vs single norm=12.2, 振幅が 30% 縮小)
```

5603 件の学習データは 272 曲の MSST 分離成果物に由来する。各曲の残留リバーブ、denoise artifact、環境差が CAMPPlus に捕捉された後で平均されるため、**録音環境ノイズまで音色へ平均混入された**。その結果、192 次元ベクトルが花丸の実際の音色方向からずれた。

### 5.3 spk_embedding の勾配がゼロ

60k 歩の学習後：

```
hanamaru_avg_style.pt vs ft_model.pth 中 spk_embedding:
  cos = 1.000000    （完全に変化なし）
  L2  = 0.0000
```

`spk_embedding` は学習に参加していたが、勾配はほぼ 0。初期化がそのまま終点だった。問題は最初から固定されていた。

### 5.4 CFM が過学習された

| モデル | CFM cos vs 事前学習 | 音色 |
|------|:--:|------|
| V1 15000歩 | 0.9996 | ✅ 音色が良い |
| exp1 12000歩 | 0.9997 | ✅ 音色が良い |
| spkemb 40000歩 | 0.921 | ❌ 重すぎる |
| spkemb 60000歩 | ~0.90 | ❌ 全体がフェイダン状態 |

CFM は誤った style 条件の下で大量の補償変換を学んでしまい、lr=1e-4 がその過程を加速した。

### 5.5 重要な検証：CAMPPlus リアルタイム vs spk_embedding

spkemb 60k の CFM 重み + `my_inference.py`（`花丸-平-voice.mp3` から CAMPPlus をリアルタイム抽出）で推論：

→ 音色は「まあまあ」回復するが、声が「厚い」

**結論**：60k CFM 自体はまだ使えるが、エネルギー均衡 loss による過学習で厚みが出ている。音色差は完全に avg_style が原因。

### 5.6 mel2 比較実験

| ソース | 効果 |
|------|------|
| `花丸-平-voice.mp3` 単一参照 | ✅ 正常 |
| 283件学習データ平均 | ❌ かなり悪い |
| 全ゼロベクトル | ❌ かなり悪い |

mel2 は拡散のアンカーであり、単純に平均化できない。平均 mel は時間方向のダイナミクス（attack/decay/ビブラート包絡）を消してしまい、ゼロベクトルはアンカーを失って出力を劣化させる。現行案（単一の高品質音声）が最適解。

### 5.7 教訓

- **CAMPPlus は録音環境に極めて敏感**であり、多ソースデータの単純平均はノイズを持ち込む
- **YingMusic 微調整では `CFM cos > 0.95` が必要**（lr≤3e-5 またはステップ数≤20k）。そうでないと音色損失が不可逆になる
- **spk_embedding の勾配は極めて弱い**ため、高品質な単一 CAMPPlus で初期化する必要があり、学習による修正を期待してはいけない

---

## 6. 4モデルマトリクス比較（8曲 · 2026-05-04）

### 6.1 比較モデル

| モデル | 学習ステップ | lr | データ量 | CFM cos | 推論方式 | 推論ステップ |
|------|:--:|:--:|------|:--:|------|:--:|
| **spkemb 60k** | 60000 | 1e-4 | 5603件 | ~0.90 | my_inference.py (CAMPPlusリアルタイム) | 50 |
| **exp1** | 12000 | 5e-6 | 1153件 | 0.9997 | my_inference.py (CAMPPlusリアルタイム) | 50 |
| **cosine** | 3000 | 1e-4 | 1153件 | 0.999999 | my_inference.py (CAMPPlusリアルタイム) | 50 |
| **v3 20k campplus** | 20000 | 3e-5 | 5603件 | 0.9960 | my_inference.py (CAMPPlusリアルタイム) | 100 |
| **V1 15000** (参照) | 15000 | 5e-6 | 1153件 | 0.9996 | inference.py (f0_cond=True) | 70 |

テスト曲：花丸カバーのドライボーカル 8曲（SummerMemory、星空、Talking、恋愛サーキュレーション など）。BVID は `矩阵对比/output/` を参照。

### 6.2 総合評価（「清涼感」順）

| モデル | 音色類似度 | 清涼感 | 厚み | 自然さ | 位置づけ |
|------|:--:|:--:|:--:|:--:|------|
| **cosine 3000** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐ | 軽快な曲の第一候補 |
| **exp1 12000** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | 清涼で柔らかい系 |
| **v3 20k campplus** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 音色精度の基準 |
| **spkemb 60k** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 特殊な厚みスタイル |
| V1 15000 (参照) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | 音色は最も似るが刺さる |

### 6.3 中核的発見：学習深度 vs 音色傾向

```
清涼 ←——————————————————————————————————————————→ 厚い/正確

cosine(3k)   exp1(12k)        v3(20k)          spkemb(60k)
↑ lr=1e-4     ↑ lr=5e-6       ↑ lr=3e-5        ↑ lr=1e-4
  CFM≈不変       CFM≈不変       CFM微変化(0.004) CFM大幅書き換え(0.10)
  データ1153     データ1153     データ5603       データ5603
```

**法則**：

1. **清涼感は CFM がほぼ動かないことから来る**：cosine（cos=0.999999）と exp1（cos=0.9997）の CFM は事前学習モデルのクリーンな質感を保持している。低 lr + 少ステップ = style_r だけを学び、mel 生成の中核に触れていない。

2. **厚みは CFM の書き換えから来る**：spkemb 60k（cos≈0.90）は、エネルギー均衡 loss が CFM mel 生成経路を過学習し、出力が「厚く」なった。v3 20k（cos=0.996）は中間で、cosine/exp1 より厚いが 60k よりかなり軽い。

3. **音色精度は正しい style + 適度な学習から来る**：v3 20k campplus は単一の高品質 CAMPPlus 初期化 + real-time CAMPPlus 推論により、音色が最も正確。60k の CFM は過学習しているが、リアルタイム CAMPPlus が音色方向を救った。

4. **データ量は決定要因ではない**：cosine/exp1 は 1153 件のみで、逆に清涼な効果が出た。v3/60k は 5603 件を使ったが、lr/ステップ数が大きすぎると厚くなる。

### 6.4 実戦での選定案

| 曲タイプ | 推奨モデル | 理由 |
|------|------|------|
| 軽快/かわいい系 | **cosine 3000** または **exp1** | 声が清涼で、厚くない |
| バラード/スローテンポ | **v3 20k campplus** | 音色が最も正確で、感情表現が良い |
| 厚み/力強さが必要 | **spkemb 60k** | 独特の厚い音色 |
| 音色優先（スタイル不問） | **v3 20k campplus** | 最も花丸に近い |
| バランス優先 | **exp1 12000** | 清涼感 + 音色がどちらも良い |

### 6.5 V1 との位置づけ

- **V1 15000** は今も音色が最も似るモデルだが、やや刺さる（V1 の均一 L1 loss は高域制約が不足）
- **YingMusic 4モデル**は全体として V1 より柔らかい（エネルギー均衡 loss の効果）。代償として音色類似度はやや劣る
- 2系統のモデルは**相互補完**：V1 は音色の極致を狙い、YingMusic は聴感の快適さを狙う
