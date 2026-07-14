# YingMusic-SVC

[GiantAILab/YingMusic-SVC](https://github.com/GiantAILab/YingMusic-SVC) オリジナル\
↓\
[rabbit321011/YingMusic-SVC-fine-tune](https://github.com/rabbit321011/YingMusic-SVC-fine-tune) ファインチューニング対応fork\
↓\
これ（とりあえずWindowsで動くようにした）


## インストール

お好みのtorchで
```
uv pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install -r requirements.txt
```

ファインチューニングする場合は[checkpoints_v2.zip](https://github.com/hetima/YingMusic-SVC/releases/download/20260714/checkpoints_v2.zip)をダウンロード、解凍し、`YingMusic-SVC/modules/openvoice/checkpoints_v2`と配置してください（あるいは任意の場所に置いて`YingMusic-SVC.yml`の記述を合わせる）。


---

--- YingMusic-SVC README 日本語訳 ---

## 概要 ✨

歌声変換（SVC）は、メロディーと歌詞を維持しながら、変換先歌手の音色で歌声を生成することを目的とします。しかし、既存のゼロショットSVCシステムは、ハーモニーの干渉、F0の誤差、歌唱に特化した帰納バイアスの不足により、実際の楽曲では不安定な場合があります。

本研究では、継続事前学習、堅牢な教師あり微調整、Flow-GRPO強化学習を統合した堅牢なゼロショットフレームワーク **YingMusic-SVC** を提案します。本モデルは、音色と内容の分離を実現する歌唱データで学習したRVC音色シフター、動的な歌唱表現に対応するF0対応音色アダプター、高域の再現性を高めるエネルギー均衡Rectified Flow Matching lossを導入しています。

複数トラックからなる難易度別ベンチマークでの実験により、YingMusic-SVCは、音色類似度、明瞭度、知覚上の自然さにおいて、強力なオープンソースベースラインを一貫して上回ることが示されました。特に伴奏やハーモニーが混在する条件で有効であり、実環境でのSVC運用に適していることが確認されています。

### 🔧 主な特徴

- **3段階の学習パイプライン**
  - **CPT**：歌唱向けモジュールを用いた継続事前学習
  - **SFT**：*F0摂動* と *ハーモニー拡張* を用いた堅牢な教師あり微調整
  - **RL（Flow-GRPO）**：知覚品質を対象とした複数報酬による強化学習

- **歌唱特有の帰納バイアス**
  - 🎼 **RVCベースの音色シフター**（120人の歌手で学習）
  - 🎚️ **F0対応のきめ細かな音色アダプター**
  - 🔊 **エネルギー均衡Flow Matching loss**（高域のディテールを強化）

---

## ニュースと更新情報 🗞️

- **2025-11-26**：伴奏分離の推論CLIとモデルチェックポイントを公開
- **2025-11-26**：簡単に試せるGradioアプリを公開
- **2025-11-25**：技術レポートを公開
- **2025-11-25**：YingMusic-SVC推論CLIの初版を公開
- **2025-11-25**：モデルチェックポイントを公開
- **2025-11-25**：マルチトラックベンチマークを公開

---

## インストール 🛠️

```bash
git clone https://github.com/GiantAILab/YingMusic-SVC.git
cd YingMusic-SVC

conda create -n ymsvc python=3.10
conda activate ymsvc
pip install -r requirements.txt

# ffmpegとsoxをインストール
sudo apt update
sudo apt install -y sox libsox-fmt-all
sudo apt install -y ffmpeg
```

---

## クイックスタート 🚀

### 1. **伴奏分離**

```bash

cd accom_separation
bash infer.sh

```

### 2. **SVC推論**

```bash
bash my_infer.sh
```

### 3. **Gradioアプリ**

```bash
python gradio_app.py
```

---

## ベンチマークデータセット 📚

100曲以上のマルチトラックスタジオ楽曲から作成した、**難易度別ベンチマーク**を提供しています。

[🤗 ダウンロード](https://huggingface.co/datasets/GiantAILab/YingMusic-SVC_Difficulty-Graded_Benchmark)  
[🔮 ダウンロード](https://www.modelscope.cn/datasets/giantailab/YingMusic-SVC_Difficulty-Graded_Benchmark)

| レベル | 説明 |
| --- | --- |
| **GT Leading** | クリーンなスタジオ収録のリードボーカル |
| **Mix Vocal** | リードボーカルとハーモニーが混在 |
| **Ours Leading** | 本プロジェクトのBand RoFormer分離モデルで抽出 |

---

## 事前学習済みモデル 🧪

| モデル | 説明 | リンク |
| --- | --- | --- |
| **YingMusic-SVC-full** | 強化学習で改善した最終モデル | [![Hugging Face](https://img.shields.io/badge/🤗%20HuggingFace-YingMusic--SVC--Full-yellow)](https://huggingface.co/GiantAILab/YingMusic-SVC/blob/main/YingMusic-SVC-full.pt) |
| **our BR separator** | 本プロジェクトの伴奏分離モデル | [![Hugging Face](https://img.shields.io/badge/🤗%20HuggingFace-BR--separator-yellow)](https://huggingface.co/GiantAILab/YingMusic-SVC/blob/main/bs_roformer.ckpt) |

---

## 開発ロードマップとTODO 🗺️

- [x] 音源分離推論CLIとモデルチェックポイント
- [x] YingMusic-SVC用Gradioアプリの開発
- [ ] ベンチマークのワンクリック評価スクリプト

---

## 謝辞 🙏

本プロジェクトは以下の成果を基盤としています。

- [Seed-VC](https://github.com/Plachtaa/seed-vc)


## 引用 🧾

YingMusic-SVCを研究で利用する場合は、以下を引用してください。

```

@article{chen2025yingmusicsvc,
  title={YingMusic-SVC: Real-World Robust Zero-Shot Singing Voice Conversion with Flow-GRPO and Singing-Specific Inductive Biases},
  author={Chen, Gongyu and Zhang, Xiaoyu and Weng, Zhenqiang and Zheng, Junjie and Shen, Da and Ding, Chaofan and Zhang, Wei-Qiang and Chen, Zihao},
  journal={arXiv preprint arXiv:2512.04793},
  year={2025}
}

```

---

## ライセンス 📝

本プロジェクトのコードはMIT Licenseのもとで公開されています。

---

--- YingMusic-SVC-fine-tune README の日本語訳 ---

# YingMusic-SVC 微調整学習コード（4種類の学習済みモデルと使用方法付き）

> 任意の歌唱音声を指定した音色へ変換します。
>
> 動作確認済み環境：RTX 5070 Ti Laptop 12GB・Win11・Python 3.10・CUDA 12.8・PyTorch 2.11.0

***

## プロジェクト概要

[YingMusic-SVC](https://github.com/GiantAILab/YingMusic-SVC) は、GiantAILab が2025年末に公開したゼロショット歌声変換プロジェクトです。Seed-VCアーキテクチャをベースに、エネルギー均衡lossや `style_residual` などの改良が加えられています。ただし、原作者は現在（2026年5月）も学習コードを公開しておらず、公式リポジトリは推論のみをサポートしています。

本プロジェクトでは、完全な微調整学習パイプラインを実装し、異なる特徴を持つ花丸の音色モデルを4種類作成しました。

***

## モデルのダウンロード

5種類のモデル重みをHuggingFaceにアップロードしています。

| モデル | ステップ数 | スタイル | 推奨用途 |
| --- | :---: | --- | --- |
| **v3 20k campplus** | 20000 | 音色が最も正確 | 音色精度の基準、バラード |
| **exp1 12000** | 12000 | 爽やかで柔らかい | バランス重視の第一候補 |
| **cosine 3000** | 3000 | 最も爽やか | 軽快・かわいい系の曲 |
| **spkemb 60k** | 60000 | 重厚 | 力強さが必要な曲 |
| **V1 15000**（Seed-VC。実際には本プロジェクトのモデルではありません） | 15000 | 最も似ている（やや刺々しい） | 音色優先（他のモデルの方が良い印象です） |

> 🔗 HuggingFaceリポジトリ：[321oll/hanamaru_hareru_YingMusicModel at main](https://huggingface.co/321oll/hanamaru_hareru_YingMusicModel/tree/main)
>
> YingMusicモデルの推論には `--target` 参照音声（`花丸-平-voice.mp3`）が必要です。推論時にCAMPPlusで音色を抽出します。
> （cosineとexp1は出力音量がやや小さくなるようです）

また、学習の開始点としてYingMusic公式の事前学習済み重みをダウンロードしてください。

```bash
# YingMusic-SVC-full.pt（約1.4GB）
huggingface-cli download GiantAILab/YingMusic-SVC YingMusic-SVC-full.pt --local-dir ./pretrained
```

***

## クイックスタート

### 環境のインストール

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers librosa soundfile pyyaml tqdm accelerate
pip install huggingface_hub
```

### 依存モデル（自動ダウンロード）

推論・学習時に、以下のモデルが `hf_hub_download` によりHuggingFaceから自動的にダウンロードされます。

- `openai/whisper-small` — 音声内容の抽出
- `funasr/campplus` — 話者の音色エンコーディング
- `lj1995/VoiceConversionWebUI` → `rmvpe.pt` — F0抽出
- `nvidia/bigvgan_v2_44khz_128band_512x` — ボコーダー

***

## 推論

### 1. my_inference.py（推奨 — CAMPPlusリアルタイム抽出、音色が最も正確）

YingMusic公式の推論スクリプトです。target音声からCAMPPlus speaker embeddingをリアルタイムに抽出します。

```bash
python my_inference.py \
    --source <入力曲.wav> \
    --target <参照音声.mp3> \
    --checkpoint output_models/spkemb_v3_lr3e-5_20k/ft_model.pth \
    --config configs/YingMusic-SVC.yml \
    --diffusion-steps 100 \
    --expname v3_20k_campplus \
    --fp16 true \
    --cuda 0
```

| 引数 | 説明 |
| --- | --- |
| `--source` | 入力する歌唱音声（任意の話者） |
| `--target` | 参照音声。`花丸-平-voice.mp3` を推奨 |
| `--diffusion-steps` | 拡散ステップ数：50（高速）〜100（高品質） |
| `--checkpoint` | モデル重みのパス |
| `--config` | `configs/YingMusic-SVC.yml` |
| `--expname` | 実験名。`./outputs/<expname>/` に出力 |

### 2. inference_spkemb.py（target不要 — spk_embedding参照方式）

自作の推論スクリプトです。音色がモデル重みに固定されているため、**targetの指定は不要**です。

> ⚠️ 実測ではCAMPPlusリアルタイム方式（`my_inference.py`）より効果が劣ります。spk_embeddingのcosが1.000でも同様です。基本的には `my_inference.py` の使用を推奨します。

```bash
python inference_spkemb.py \
    --source <入力曲.wav> \
    --checkpoint output_models/yingmusic_spkemb_60k/ft_model.pth \
    --config configs/YingMusic-SVC.yml \
    --diffusion-steps 50 \
    --inference-cfg-rate 0.7 \
    --output ./output.wav
```

初回実行時に `花丸-平-voice.mp3` から mel2 と `S_ori` を抽出し、キャッシュします。

### 3. inference.py（Seed-VC V1）

V1推論では、target参照音声と `f0_condition` が必要です。

```bash
python inference.py \
    --source <入力曲.wav> \
    --target <花丸の参照音声.mp3> \
    --checkpoint runs/hanamaru_full_fav_step15000/ft_model.pth \
    --config configs/my_finetune_12g.yml \
    --diffusion-steps 20 \
    --inference-cfg-rate 0.7 \
    --f0-condition true \
    --fp16 true
```

***

## 学習

### データの準備

学習データは花丸晴琉のボーカルのみの音声です（MSSTで分離後、15〜30秒の音声に分割）。

```
train_data/
├── speaker1/         # 1153個、約7.2時間（初期93曲）
└── speaker1-plus/    # 4450個、約29.2時間（追加262曲）
```

分割ツール：`recut.py`

```bash
python recut.py --input_dir <MSST生成物のディレクトリ> --output_dir train_data/speaker1-plus
```

### 学習コマンド

#### spk_embedding方式（主力、5603個のデータ）

```bash
accelerate launch --config_file accelerate_config.yaml \
    train_yingmusic_ft_spkemb.py \
    --config configs/YingMusic-SVC.yml \
    --pretrained-ckpt pretrained/YingMusic-SVC-full.pt \
    --dataset-dir train_data \
    --run-name yingmusic_spkemb \
    --batch-size 1 \
    --max-steps 60000 \
    --save-every 10000
```

#### v2方式（環境変数によるlr/initの注入に対応、v3 20kで使用）

```bash
# CAMPPlusを1音声から初期化、lr=3e-5、20kステップ
$env:SPKEMB_V2_STYLE_PATH = "output_models/hanamaru_prompt_style.pt"
$env:SPKEMB_V2_BASE_LR = "3e-5"
$env:SPKEMB_V2_WARMUP = "3000"
$env:SPKEMB_V2_LR_MIN = "1e-6"

accelerate launch --config_file accelerate_config.yaml \
    _train_spkemb_v2.py \
    --config configs/YingMusic-SVC.yml \
    --pretrained-ckpt pretrained/YingMusic-SVC-full.pt \
    --dataset-dir train_data \
    --run-name spkemb_v3_lr3e-5_20k \
    --batch-size 1 \
    --max-steps 20000 \
    --save-every 5000
```

#### 基本方式（lr=5e-6、exp1で使用）

```bash
accelerate launch --config_file accelerate_config.yaml \
    train_yingmusic_ft.py \
    --config configs/YingMusic-SVC.yml \
    --pretrained-ckpt pretrained/YingMusic-SVC-full.pt \
    --dataset-dir train_data \
    --run-name yingmusic_exp1 \
    --batch-size 1 \
    --max-steps 12000 \
    --save-every 2000
```

#### Cosine方式（lr=1e-4、warmupあり、cosine 3kで使用）

```bash
accelerate launch --config_file accelerate_config.yaml \
    train_yingmusic_ft_cosine.py \
    --config configs/YingMusic-SVC.yml \
    --pretrained-ckpt pretrained/YingMusic-SVC-full.pt \
    --dataset-dir train_data \
    --run-name yingmusic_cosine \
    --batch-size 1 \
    --max-steps 3000 \
    --save-every 500
```

### 主要ハイパーパラメータの推奨値

| パラメータ | 推奨値 | 説明 |
| --- | :---: | --- |
| base_lr | 3e-5 〜 5e-6 | lrが低いほど「爽やか」に、高いほど「重厚」になる |
| 総ステップ数 | 3k 〜 20k | CFM cosが0.95を超えると安全圏 |
| warmup | 3000 | CosineWarmupScheduler |
| batch_size | 1 | 12GB VRAMでの上限 |
| spk初期化 | **音声1個分のクリーンなCAMPPlus** | 複数ソースの平均は絶対に使用しない |

> ⚠️ **学習で得られた重要な知見**：`spk_embedding` の勾配はほぼ0で、初期化品質が最終品質を決めます。CFM cosが高いほど爽やかに、低いほど重厚になります。詳しくは `docs/YingMusic微調整の補足.md` の5〜6節を参照してください。

***

## 4モデルの比較

花丸のカバー曲8曲のボーカルのみ音声で比較を行いました。

```
爽やか ←————————————————————————————————————→ 重厚／正確

cosine(3k)   exp1(12k)       v3(20k)        spkemb(60k)
最も爽やか    爽やかで柔らかい   中間            最も重厚
CFM変化なし   CFM変化なし       CFM微変化       CFM大幅変更
cos=1.0       cos=0.9997       cos=0.996       cos≈0.90
```

### モデルの選び方

| 用途 | モデル | diff steps |
| --- | --- | :---: |
| 花丸に最も似せる（聴感は問わない） | V1 15000 | 20 |
| 最も正確で自然な音色 | v3 20k campplus | 100 |
| 爽やかさ重視 | cosine 3000 | 50 |
| バランス重視 | exp1 12000 | 50 |
| 重厚で力強い音色 | spkemb 60k | 50 |

> 詳細な分析は `docs/モデル比較と最終選定.md` と `docs/YingMusic微調整の補足.md` を参照してください。

***

## ファイル構成

本リポジトリは [GiantAILab/YingMusic-SVC](https://github.com/GiantAILab/YingMusic-SVC) のforkで、**完全な微調整学習機能を追加しています**。以下に注釈付きのディレクトリ構成を示します。

```
YingMusic-SVC/                        # GiantAILab/YingMusic-SVCからfork
│
│  ── 元リポジトリのファイル（保持） ──
├── my_inference.py                   # ★ 公式推論（CAMPPlusリアルタイム）
├── my_infer.sh                       # 公式推論のショートカットスクリプト
├── gradio_app.py                     # Gradio WebUI
├── mm4.py                            # 音声前処理
├── hf_utils.py                       # HuggingFaceモデルのダウンロード
├── requirements.txt                  # 元リポジトリの依存関係
├── Remix/                            # リバーブ・エコーの後処理
├── accom_separation/                 # 伴奏分離（Band RoFormer）
├── utils/                            # ログユーティリティ
│
│  ── 追加：学習コード ──
├── train_yingmusic_ft_spkemb.py      # ★ 主な学習スクリプト（spk_embedding方式）
├── _train_spkemb_v2.py               # v2学習スクリプト（環境変数によるlr/init注入）
├── train_yingmusic_ft.py             # 基本学習（lr=5e-6）
├── train_yingmusic_ft_cosine.py      # Cosine学習（lr=1e-4、warmupあり）
├── optimizers.py                     # オプティマイザの構築
├── optimizers_cosine.py              # CosineWarmupScheduler
├── accelerate_config.yaml            # accelerate分散設定
├── data/
│   └── ft_dataset.py                 # 学習データローダー
│
│  ── 追加：推論 + モジュール ──
├── inference_spkemb.py               # target不要の推論（spk_embedding参照）
├── inference.py                      # Seed-VC V1推論（追加機能、YingMusic標準ではない）
├── modules/openvoice/                # OpenVoice音色揺らぎ（学習用、Pythonのみ）
├── 花丸-平-voice.mp3                 # 花丸の参照音声
│
│  ── 追加：ドキュメント + ツール ──
├── docs/
│   ├── Seed-VC-YingMusic技術全体像と微調整ロードマップ.md
│   ├── YingMusic微調整の補足.md        # 学習履歴 + 4モデルの比較
│   └── モデル比較と最終選定.md            # 3モデルの評価 + 選定早見表
├── configs/
│   └── my_finetune_12g.yml           # Seed-VC V1微調整バランス設定（YingMusic-SVC.ymlは元から存在）
├── recut.py                          # ボーカル音声の分割ツール
├── extract_avg_campplus.py           # CAMPPlus平均値の抽出
├── _check_v3_weights.py              # 重みの偏移検出
├── batch_matrix_infer.py             # 一括マトリクス推論
├── clean_silence.py                  # 無音処理
└── .gitignore                        # 重み・データ・キャッシュを除外
```

> ℹ️ ルートディレクトリには `inference.py` が2つあります。`inference.py`（追加したSeed-VC V1推論）と `accom_separation/inference.py`（元リポジトリの伴奏分離推論）は別のファイルであり、互いに干渉しません。

***

## よくある質問

**Q: どの推論スクリプトが最もよいですか？**
A: `my_inference.py`（CAMPPlusリアルタイム）> `inference_spkemb.py`（spk_embedding参照）です。spk_embedding cosが1.000でも、リアルタイムCAMPPlusの方が聴感は良好です。

**Q: 学習結果の音色がおかしいのはなぜですか？**
A: 99%は `spk_embedding` の初期化が原因です。クリーンな参照音声から抽出した、品質の高いCAMPPlusを1音声分だけ使用してください。複数ソースの平均は絶対に使用しないでください。

**Q: 爽やかさと音色の正確さはどう両立すればよいですか？**
A: 完全な両立はできません。CFM cosが高いほど爽やか（lrが低い／ステップ数が少ない）になり、低いほど重厚ですが音色がより強く反映されます。4種類のモデルを試し、曲調に応じて選ぶことを推奨します。

**Q: VRAM 12GBで足りますか？**
A: 学習は `batch_size=1` でぎりぎり（約11GB）です。推論は100ステップでも問題ありません。

**Q: 別のキャラクターのデータで学習できますか？**
A: 可能です。`train_data/` 以下の音声ファイルを置き換え、`spk_embedding` の初期化用として高品質なCAMPPlusを1音声分用意してください。

**Q: データセットには何が必要ですか？**
A: 実測では5時間程度のデータでも十分な効果がありました。5時間未満は未検証です。微調整後の推論では、クリーンで明瞭な参照音声を使用することを推奨します。また、微調整だけで1人の複数の声質を同時に再現するのは難しい場合があるため、声質ごとに分けて学習する必要があるかもしれません。

***

## 謝辞

- [GiantAILab/YingMusic-SVC](https://github.com/GiantAILab/YingMusic-SVC) — 基本アーキテクチャと事前学習済み重み
- [Plachtaa/Seed-VC](https://github.com/Plachtaa/Seed-VC) — 上流アーキテクチャとOpenVoiceモジュール
- [myshell-ai/OpenVoice](https://github.com/myshell-ai/OpenVoice) — 学習時の音色揺らぎ
- 花丸晴琉 — 学習データの提供（花丸が一番かわいい！）
- 花丸の音声を切り出してくれた皆さん、本当にありがとうございます
- データはcyanAIがBilibiliから取得したものです（cyanAIは私の別プロジェクトです）
- deepseek v4proの安価なトークン

***

## ライセンス

本プロジェクトのコードはMIT Licenseに基づきます。モデル重みは研究・個人利用のみを目的としています。
