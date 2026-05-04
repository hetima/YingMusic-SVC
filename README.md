# YingMusic-SVC 微调训练代码(附带有四个实例模型以及其使用方法)

> 将任意唱歌音频转换为指定音色
>
> 验证过的环境配置：RTX 5070 Ti Laptop 12GB · Win11 · Python 3.10 · CUDA 12.8 · PyTorch 2.11.0

***

## 项目背景

[YingMusic-SVC](https://github.com/GiantAILab/YingMusic-SVC) 是 GiantAILab 于 2025 年底开源的零样本歌声转换项目，基于 Seed-VC 架构扩展，增加了能量均衡 loss、style\_residual 等创新。**但原作者至今（2026-05）未开源训练代码**，官方仓库仅支持 inference。

本项目**从零编写了完整的微调训练管线**，经过三轮实验迭代，产出 4 个可供使用的花丸音色模型（各有风格侧重），覆盖从"清新轻快"到"厚重准确"的完整风格谱系。

***

## 模型下载

5 个模型权重已上传至 HuggingFace：

| 模型                     |   步数  | 风格        | 推荐场景        |
| ---------------------- | :---: | --------- | ----------- |
| **v3 20k campplus**    | 20000 | 音色最准      | 音色准度标杆，抒情慢歌 |
| **exp1 12000**         | 12000 | 清新柔和      | 平衡首选        |
| **cosine 3000**        |  3000 | 最清新       | 轻快/可爱系歌曲    |
| **spkemb 60k**         | 60000 | 厚重        | 需要力量感的歌曲    |
| **V1 15000** (Seed-VC，这个其实不属于本项目，只是附上) | 15000 | 音色最像（偏刺耳） | 音色优先(感觉音色其实不如其他几个)        |

> 🔗 HuggingFace 仓库：[321oll/hanamaru\_hareru\_YingMusicModel at main](https://huggingface.co/321oll/hanamaru_hareru_YingMusicModel/tree/main)
>
> 所有 YingMusic 模型推理时需要 `--target` 参考音频（花丸-平-voice.mp3），模型通过实时 CAMPPlus 提取音色。
（貌似cosine和exp1的输出音量会偏小）
还需下载 YingMusic 官方预训练权重作为训练起点：

```bash
# YingMusic-SVC-full.pt（约 1.4GB）
huggingface-cli download GiantAILab/YingMusic-SVC YingMusic-SVC-full.pt --local-dir ./pretrained
```

***

## 快速开始

### 环境安装

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers librosa soundfile pyyaml tqdm accelerate
pip install huggingface_hub
```

### 依赖模型（自动下载）

推理/训练时以下模型会通过 `hf_hub_download` 自动从 HuggingFace 下载：

- `openai/whisper-small` — 语义内容提取
- `funasr/campplus` — 说话人音色编码
- `lj1995/VoiceConversionWebUI` → `rmvpe.pt` — F0 提取
- `nvidia/bigvgan_v2_44khz_128band_512x` — 声码器

***

## 推理

### 1. my\_inference.py（推荐 — CAMPPlus 实时，音色最准）

YingMusic 官方推理脚本，从 target 音频实时提取 CAMPPlus speaker embedding。

```bash
python my_inference.py \
    --source <输入歌曲.wav> \
    --target <花丸参考音频.mp3> \
    --checkpoint output_models/spkemb_v3_lr3e-5_20k/ft_model.pth \
    --config configs/YingMusic-SVC.yml \
    --diffusion-steps 100 \
    --expname v3_20k_campplus \
    --fp16 true \
    --cuda 0
```

| 参数                  | 说明                             |
| ------------------- | ------------------------------ |
| `--source`          | 输入唱歌音频（任意人）                    |
| `--target`          | 花丸参考音频，推荐 `花丸-平-voice.mp3`     |
| `--diffusion-steps` | 扩散步数：50（快速）\~ 100（高质量）         |
| `--checkpoint`      | 模型权重路径                         |
| `--config`          | `configs/YingMusic-SVC.yml`    |
| `--expname`         | 实验名，输出到 `./outputs/<expname>/` |

### 2. inference\_spkemb.py（无 target — spk\_embedding 查表）

我们的自写推理脚本，音色固化在模型权重内，**不需要传 target**。

> ⚠️ 实测效果不如 CAMPPlus 实时（my\_inference.py），即使 spk\_embedding cos=1.000。推荐优先使用 my\_inference.py。

```bash
python inference_spkemb.py \
    --source <输入歌曲.wav> \
    --checkpoint output_models/yingmusic_spkemb_60k/ft_model.pth \
    --config configs/YingMusic-SVC.yml \
    --diffusion-steps 50 \
    --inference-cfg-rate 0.7 \
    --output ./output.wav
```

首次运行自动从 `花丸-平-voice.mp3` 提取 mel2 + S\_ori 并缓存。

### 3. inference.py（Seed-VC V1）

V1 推理，需要 target 参考音频 + f0\_condition。

```bash
python inference.py \
    --source <输入歌曲.wav> \
    --target <花丸参考音频.mp3> \
    --checkpoint runs/hanamaru_full_fav_step15000/ft_model.pth \
    --config configs/my_finetune_12g.yml \
    --diffusion-steps 20 \
    --inference-cfg-rate 0.7 \
    --f0-condition true \
    --fp16 true
```

***

## 训练

### 数据准备

训练数据为花丸晴琉的干声片段（MSST 分离 → 切分为 15-30s 片段）：

```
train_data/
├── speaker1/         # 1153 条，~7.2h（早期 93 首）
└── speaker1-plus/    # 4450 条，~29.2h（新增 262 首）
```

切分工具：`recut.py`

```bash
python recut.py --input_dir <MSST产物目录> --output_dir train_data/speaker1-plus
```

### 训练命令

#### spk\_embedding 方案（主力，5603 条数据）

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

#### v2 方案（支持环境变量注入 lr/init，v3 20k 所用）

```bash
# 单条 CAMPPlus 初始化，lr=3e-5，20k 步
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

#### 基础方案（lr=5e-6，exp1 所用）

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

#### Cosine 方案（lr=1e-4 warmup，cosine 3k 所用）

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

### 关键超参建议

| 参数          |        推荐值        | 说明                    |
| ----------- | :---------------: | --------------------- |
| base\_lr    |    3e-5 \~ 5e-6   | lr 越低越"清新"，越高越"厚重"    |
| 总步数         |     3k \~ 20k     | CFM cos > 0.95 为安全区   |
| warmup      |        3000       | CosineWarmupScheduler |
| batch\_size |         1         | 12GB 显存的极限            |
| spk 初始化     | **单条干净 CAMPPlus** | 绝对不要用多源平均！            |

> ⚠️ **训练核心教训**：spk\_embedding 梯度 ≈ 0，初始化质量 = 最终质量。CFM cos 越高越清新，越低越厚重。详见 `docs/对Yingmusic微调的补充.md` 第 5-6 节。

***

## 四模型对比

在 8 首花丸翻唱干声上完成矩阵对比：

```
清新 ←————————————————————————————————————→ 厚重/准确

cosine(3k)   exp1(12k)       v3(20k)        spkemb(60k)
最清新         清新柔和         居中            最厚重
CFM不动        CFM不动         CFM微动         CFM大改
cos=1.0       cos=0.9997      cos=0.996       cos≈0.90
```

### 选型建议

| 场景         | 模型              | diff steps |
| ---------- | --------------- | :--------: |
| 最像花丸（不管听感） | V1 15000        |     20     |
| 音色最准 + 自然  | v3 20k campplus |     100    |
| 清新首选       | cosine 3000     |     50     |
| 平衡之选       | exp1 12000      |     50     |
| 厚重力量感      | spkemb 60k      |     50     |

> 详细分析见 `docs/模型对比与最终选型.md` 和 `docs/对Yingmusic微调的补充.md`。

***

## 文件结构

本仓库是 [GiantAILab/YingMusic-SVC](https://github.com/GiantAILab/YingMusic-SVC) 的 fork，**新增了完整的微调训练支持**。以下是带标注的完整目录：

```
YingMusic-SVC/                        # fork 自 GiantAILab/YingMusic-SVC
│
│  ── 原仓库文件（保留） ──
├── my_inference.py                   # ★ 官方推理（CAMPPlus 实时）
├── my_infer.sh                       # 官方推理快捷脚本
├── gradio_app.py                     # Gradio WebUI
├── mm4.py                            # 音频预处理
├── hf_utils.py                       # HuggingFace 模型下载
├── requirements.txt                  # 原仓库依赖
├── Remix/                            # 混响/回声后处理
├── accom_separation/                 # 伴奏分离（Band RoFormer）
├── utils/                            # 日志工具
│
│  ── 新增：训练代码 ──
├── train_yingmusic_ft_spkemb.py      # ★ 主训练脚本（spk_embedding 方案）
├── _train_spkemb_v2.py               # v2 训练脚本（环境变量注入 lr/init）
├── train_yingmusic_ft.py             # 基础训练（lr=5e-6）
├── train_yingmusic_ft_cosine.py      # Cosine 训练（lr=1e-4 warmup）
├── optimizers.py                     # 优化器构建
├── optimizers_cosine.py              # CosineWarmupScheduler
├── accelerate_config.yaml            # accelerate 分布式配置
├── data/
│   └── ft_dataset.py                 # 训练数据加载器
│
│  ── 新增：推理 + 模块 ──
├── inference_spkemb.py               # 无 target 推理（spk_embedding 查表）
├── inference.py                      # Seed-VC V1 推理（附加，非 YingMusic 原生）
├── modules/openvoice/                # OpenVoice 音色扰动（训练用，仅 py）
├── 花丸-平-voice.mp3                 # 花丸参考音频
│
│  ── 新增：文档 + 工具 ──
├── docs/
│   ├── Seed-VC-YingMusic-技术全览与微调路线.md
│   ├── 对Yingmusic微调的补充.md        # 训练历史 + 四模型矩阵对比
│   └── 模型对比与最终选型.md            # 三模型评分 + 选型速查
├── configs/
│   └── my_finetune_12g.yml           # Seed-VC V1 微调配平（YingMusic-SVC.yml 原已存在）
├── recut.py                          # 干声切分工具
├── extract_avg_campplus.py           # CAMPPlus 均值提取
├── _check_v3_weights.py              # 权重偏移检测
├── batch_matrix_infer.py             # 批量矩阵推理
├── clean_silence.py                  # 静音处理
└── .gitignore                        # 排除权重/数据/缓存
```

> ℹ️ 根目录有两个 `inference.py`：`inference.py`（我们加的 Seed-VC V1 推理）和 `accom_separation/inference.py`（原仓库的伴奏分离推理），互不冲突。

***

## 常见问题

**Q: 用哪个推理脚本最好？**
A: `my_inference.py`（CAMPPlus 实时）> `inference_spkemb.py`（spk\_embedding 查表）。即使 spk\_embedding cos=1.000，实时 CAMPPlus 听感仍更好。

**Q: 为什么我的训练结果音色不对？**
A: 99% 是 spk\_embedding 初始化问题。必须用单条高质量 CAMPPlus（从干净参考音频提取），绝不能用多源平均。

**Q: 清新感和音色准度怎么取舍？**
A: 不可兼得。CFM cos 越高越清新（lr 越低/步数越少），CFM cos 越低越厚重但音色更融入。建议四种模型都试试，按歌曲风格选用。

**Q: 12GB 显存够吗？**
A: 训练 batch\_size=1 刚好（\~11GB），推理 100 步也安全。

**Q: 可以用其他角色的数据训练吗？**
A: 可以。替换 `train_data/` 下的音频文件，准备单条高质量 CAMPPlus 作为 `spk_embedding` 初始化即可。

**Q： 数据集要求是什么？**
A: 测试下来，其实5h的数据效果已经不错了，小于5h的还没试。建议微调后推理的参考音频用干净，清晰的。以及，微调可能没办法同时表现一个人的多个声线，可能还是得分开训练。
***

## 致谢

- [GiantAILab/YingMusic-SVC](https://github.com/GiantAILab/YingMusic-SVC) — 基础架构和预训练权重
- [Plachtaa/Seed-VC](https://github.com/Plachtaa/Seed-VC) — 上游架构和 OpenVoice 模块
- [myshell-ai/OpenVoice](https://github.com/myshell-ai/OpenVoice) — 训练时音色扰动
- 花丸晴琉 — 训练数据来源（花丸最可爱了！）
- 愿意搬运花丸切片的大家，太感谢了
- 数据来源是让cyanAI上B站扒的(cyanAI是我的另一个项目)
- deepseek v4pro的便宜token

***

## License

本项目代码基于 MIT License。模型权重仅用于研究/个人使用。
