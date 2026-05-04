# Seed-VC / YingMusic-SVC 技术全览与微调路线

> 基于多轮对话与 Grok 讨论整理 | 2026-05

---

## 目录

1. [项目背景与训练历史](#1-项目背景与训练历史)
2. [Seed-VC V1 架构全解析](#2-seed-vc-v1-架构全解析)
3. [V1 训练流程逐步拆解](#3-v1-训练流程逐步拆解)
4. [Seed-VC V2 架构与 AR 失败分析](#4-seed-vc-v2-架构与-ar-失败分析)
5. [V2 推理中 target 参考音频的作用](#5-v2-推理中-target-参考音频的作用)
6. [YingMusic-SVC vs V1 逐模块对比](#6-yingmusic-svc-vs-v1-逐模块对比)
7. [RVC Timbre Shifter 现状与替代方案](#7-rvc-timbre-shifter-现状与替代方案)
8. [微调 YingMusic 的执行路线](#8-微调-yingmusic-的执行路线)

---

## 1. 项目背景与训练历史

### 目标
将 natori_sana 的说话/唱歌音频转换为花丸（hanamaru）的音色，并保留花丸的说话风格/演唱表现力。

### 已有训练历史

| 训练名称 | 模型版本 | 数据集 | 步数 | lr | 状态 |
|------|:---:|------|:--:|:--:|:--:|
| 花丸PRUE-small | V1 | ~50 条干声 | 1750 | 1e-5 | 早期测试 |
| my_run | V1 | 少量测试数据 | 1750 | 1e-5 | 测试 |
| hanamaru_full_fav | V1 | 1153 条（93首切分） | 5500 | 1e-5 | 正式V1 |
| hanamaru_full_fav_step15000 | V1 | 同上 | +9500 (to 15000) | 5e-6 | ✅ V1 最佳 |
| hanamaru_v2_cfm | V2 (CFM only) | 花丸干声 | 6000 | — | CFM 单独训 |
| hanamaru_v2_cfm_ar | V2 (CFM+AR) | 花丸干声 | 6000 | — | ❌ AR 跨说话人失败 |
| yingmusic_exp1 | YingMusic | 1153 条 | 12000 | 5e-6 | ❌ lr太低, 几乎没学到 |
| yingmusic_cosine | YingMusic | 1153 条 | 3000 | 1e-4 warmup | ✅ 单听有进步 |
| RVC | RVC | 1153 条 | 100轮 | — | ❌ 不如 Seed-VC V1 |
| **yingmusic_spkemb_60k** | **YingMusic+spk_emb** | **5603条 / 36.4h** | **60000** | **1e-4 cosine** | **❌ 费丹→存档（有独特厚重感）** |
| **spkemb_v3_lr3e-5_20k** | **YingMusic+spk_emb_v2** | **5603条 / 36.4h** | **20000** | **3e-5 cosine** | **✅ v3 主力 (warmup 3k)** |

### 三家对比（SummerMemory 同一首歌）

| 模型 | 音色 | 柔和度 | 清新感 | 备注 |
|------|:---:|:---:|:---:|------|
| Seed-VC V1 15000步 | 最像花丸 | 刺耳 | ⭐⭐⭐ | 均匀L1 loss, 高频欠约束 |
| YingMusic exp1 12000步 (lr=5e-6) | 柔和 | 太柔→清新 | ⭐⭐⭐⭐ | lr太低, 权重几乎没变, CFM cos=0.9997 |
| YingMusic cosine 3000步 (lr=1e-4) | 有进步 | 柔和→清新 | ⭐⭐⭐⭐⭐ | cosine warmup, CFM cos=0.999999 |
| YingMusic spkemb 60k | — | —→厚重 | ⭐⭐ | avg_style 初始化导致音色偏离, CAMPPlus实时可抢救 |
| **YingMusic v3 20k campplus** | **最准** | **居中** | **⭐⭐⭐** | **lr=3e-5, 100 diff, CFM cos=0.996** |

### 当前可用模型

| 模型 | 路径 | 用途 |
|------|------|------|
| V1 lr=5e-6 15000步 | `runs/hanamaru_full_fav_step15000/ft_model.pth` | 唱歌+说话主力 |
| V2 CFM-only | `runs/hanamaru_v2_cfm/CFM_epoch_00003_step_06000.pth` | V2 说话 CC FM |
| YingMusic 预训练 | `YingMusic-SVC/YingMusic-SVC-full.pt` | 零样本推理 |
| YingMusic exp1 12000步 | `temp/temp_0502/output_models/yingmusic_exp1/` | **★ 清新柔和派** |
| YingMusic cosine 3000步 | `temp/temp_0502/output_models/yingmusic_cosine/` | **★ 最清新** |
| **YingMusic spkemb 60k** | **`temp/temp_0502/output_models/yingmusic_spkemb_60k/`** | **存档（厚重风格）** |
| **YingMusic v3 20k** | **`temp/temp_0502/output_models/spkemb_v3_lr3e-5_20k/`** | **✅ 音色最准** |
| hanamaru_avg_style.pt | `temp/temp_0502/output_models/hanamaru_avg_style.pt` | ❌ 平均 CAMPPlus (已证无效) |
| **hanamaru_prompt_style.pt** | **`temp/temp_0502/output_models/hanamaru_prompt_style.pt`** | **单条 CAMPPlus (v3 初始化)** |
| OpenVoice se_db | `seed-vc/modules/openvoice/checkpoints_v2/converter/se_db.pt` | Timbre Shifter 依赖 (100004人) |

---

## 9. 训练数据集

### 数据流水线

```
bilibili 下载的 .wav (272首)
  → batch_msst.ps1 (duality_v2 → fused dereverb → denoise)
    → 03_final/*_Vocals_dry_dry.wav (262首, 29.2h)
      → recut.py (静音检测切15-30s + silenceremove去首尾静音)
        → speaker1-plus/ (4507段, 29.2h)
          + speaker1/ (1153段, ~7.2h)
            = train_data/ (5603段, ~36.4h 净人声)
```

### 数据集统计

| 目录 | 文件数 | 时长 | 来源 |
|------|:--:|:--:|------|
| `train_data/speaker1/` | 1153 | ~7.2h | 早期 93 首切分 |
| `train_data/speaker1-plus/` | 4450 | ~29.2h | 新增 262 首切分 |
| **训练总计** | **5603** | **~36.4h** | — |
| 过期/超短被 `ft_dataset.py` 自动跳过 | ~60 | — | max=30s, min=1s |

### 切分逻辑 (recut.py)

```
输入: 03_final/*_Vocals_dry_dry.wav (只取干声, 跳过 _other)

切分策略:
  ≤30s → 整首保留
  >30s → 在15-30s窗口内找静音点切开
    优先级: 长停顿≥2s > 自然段尾≥22s > 句子间隙≥15s > 硬切29.5s

每段切出后: ffmpeg silenceremove (-35dB, 0.3s) 去首尾静音
最终检查: <1s 删除, >31s 由训练时 dataloader 自动跳过
```

---

## 10. LR Schedule

### 60k 步 Cosine Warmup

```
lr(t):
  0 ~ 3000步:    linear warmup  0 → 1e-4
  3000 ~ 60000步: cosine decay  1e-4 ↘ 1e-6
```

| 步数 | lr | 阶段 |
|:--:|:--:|------|
| 0 | 0 | 起始 |
| 3k | 1e-4 | warmup 结束 |
| 15k | 7.5e-5 | 主力学习 |
| 30k | 5e-5 | 中期收敛 |
| 45k | 2.5e-5 | 精调 |
| 60k | 1e-6 | 结束 |

对比旧方案(ExponentialLR gamma=0.999996): 15000步后 lr 仍为初始值的94%, 基本恒lr。

可视化: `temp/temp_0502/lr_schedule_60k.html`

---

## 11. Speaker Embedding 固化方案

### 目标

推理时不依赖 target 参考音频，花丸音色完全由模型权重内部表示。

### 实现 (train_yingmusic_ft_spkemb.py)

**训练时:**
```python
self.spk_embedding = nn.Embedding(1, 192)   # 花丸 = id 0
# ⚠️ avg_style (5603条均值) 初始化已证无效 — cos=0.72 vs 单条, 梯度=0 无法修正
# ✅ 改用单条高质量参考 CAMPPlus (hanamaru_prompt_style.pt)
style = self.spk_embedding(torch.zeros(B, dtype=torch.long, device=self.device))
# spk_embedding 参与 backward, 但梯度极弱, 初始化质量决定一切
# CFM + LengthRegulator + spk_embedding 三者联合优化
```

**推理时 (inference_spkemb.py):**
```python
style = spk_emb(torch.zeros(1, dtype=torch.long, device=device))
# mel2 + S_ori 从花丸参考一次提取、永久缓存
# 推理命令行不需要 --target 参数
```

### 推理命令 (无 target 音频)

```powershell
cd E:\AIscene\AISVCs\temp\temp_0502
..\..\.venv\Scripts\python.exe inference_spkemb.py `
    --source <任意音频> `
    --checkpoint output_models\yingmusic_spkemb_60k\ft_model.pth `
    --config "E:\AIscene\AISVCs\YingMusic-SVC\configs\YingMusic-SVC.yml" `
    --diffusion-steps 50
```

首次运行会从 `花丸-平-voice.mp3` 提取 mel2 和 S_ori 缓存到 `output_models/ref_mel2.pt` + `output_models/ref_S_ori.pt`，之后秒加载。

---

## 12. 训练参数 (version: yingmusic_spkemb_60k)

### 运行命令

```powershell
cd E:\AIscene\AISVCs\temp\temp_0502
..\..\.venv\Scripts\python.exe -m accelerate.commands.launch `
    --config_file accelerate_config.yaml `
    train_yingmusic_ft_spkemb.py `
    --config "E:\AIscene\AISVCs\YingMusic-SVC\configs\YingMusic-SVC.yml" `
    --pretrained-ckpt "E:\AIscene\AISVCs\YingMusic-SVC\YingMusic-SVC-full.pt" `
    --dataset-dir train_data `
    --run-name yingmusic_spkemb_60k `
    --save-every 10000
```

### 训练配置一览

| 参数 | 值 |
|------|:--:|
| 数据集 | speaker1 + speaker1-plus = 5603条 / 36.4h |
| 起点 | YingMusic-SVC-full.pt (CFM + LengthRegulator) |
| base_lr | 1e-4 |
| schedule | warmup 3k → cosine decay 60k |
| lr_min | 1e-6 |
| batch_size | 1 (12GB显存限制) |
| 速度 | ~1.1 it/s |
| 总步数 | 60000 (~7.2 epochs) |
| 预计时间 | ~15 小时 |
| 保存频率 | 每 10000 步 |
| Timbre Shifter | OpenVoice se_db.pt (100004人) |
| Speaker Embedding | nn.Embedding(1,192), avg_style初始化 |
| 能量均衡loss | ✅ balance_loss=True |
| Style Residual | ✅ return_style_residual=True |

### 输出模型

| 步数 | 文件 |
|:--:|------|
| 10000 | `DiT_epoch_*_step_10000.pth` |
| 20000 | `DiT_epoch_*_step_20000.pth` |
| 30000 | `DiT_epoch_*_step_30000.pth` |
| 40000 | `DiT_epoch_*_step_40000.pth` |
| 50000 | `DiT_epoch_*_step_50000.pth` |
| 60000 | `ft_model.pth` (最终) |

---

## 13. 关键文件清单

### 训练脚本

| 文件 | 路径 | 作用 |
|------|------|------|
| `train_yingmusic_ft_spkemb.py` | `temp/temp_0502/` | **当前 60k 训练脚本（含 spk_embedding）** |
| `train_yingmusic_ft_cosine.py` | `temp/temp_0502/` | 3000 步 cosine 实验脚本 |
| `train_yingmusic_ft.py` | `temp/temp_0502/` | 最初 lr=5e-6 实验脚本 |
| V1 训练脚本 | `seed-vc/train.py` | V1 基线 |
| V1 微调脚本 | `seed-vc/train_lr_5e-6.py` | V1 15000 步训练 |

### 推理脚本

| 文件 | 路径 | 作用 |
|------|------|------|
| `inference_spkemb.py` | `temp/temp_0502/` | **无 target 推理（固化 spk_embedding）** |
| `my_inference.py` | `YingMusic-SVC/` | YingMusic 官方推理（需 target） |
| `inference.py` | `seed-vc/` | V1 推理 |
| `inference_v2.py` | `seed-vc/` | V2 推理 |

### 数据工具

| 文件 | 路径 | 作用 |
|------|------|------|
| `batch_msst.ps1` | `cyanAIWorkAria/` | MSST 三阶段人声提取 |
| `recut.py` | `temp/temp_0502/` | 干声切分 + silenceremove |
| `extract_avg_campplus.py` | `temp/temp_0502/` | 提取平均 CAMPPlus (hanamaru_avg_style.pt) |
| `smart_cut.py` | `seed-vc/` | 原始切分脚本 |

### 调度与配置

| 文件 | 路径 | 作用 |
|------|------|------|
| `optimizers_cosine.py` | `temp/temp_0502/` | CosineWarmupScheduler |
| `lr_schedule_60k.html` | `temp/temp_0502/` | LR 曲线可视化 |
| `accelerate_config.yaml` | `temp/temp_0502/` | accelerate 配置 |
| `run_all.ps1` | `temp/temp_0502/` | 全自动 MSST→切分→训练 |

### YingMusic 核心模块

| 文件 | 路径 |
|------|------|
| `length_regulator.py` | `temp/temp_0502/modules/` (YingMusic版, 238行) |
| `flow_matching.py` | `temp/temp_0502/modules/` (YingMusic版, 667行, 含能量均衡loss) |
| `diffusion_transformer.py` | `temp/temp_0502/modules/` (YingMusic版, 564行, 含style_r通道) |
| `YingMusic-SVC.yml` | `YingMusic-SVC/configs/` |
| `YingMusic-SVC-full.pt` | `YingMusic-SVC/` |

### 训练产物

| 路径 | 内容 |
|------|------|
| `temp/temp_0502/output_models/yingmusic_spkemb_60k/` | **当前训练输出** |
| `temp/temp_0502/output_models/hanamaru_avg_style.pt` | 平均 CAMPPlus (192维) |
| `temp/temp_0502/output_models/ref_mel2.pt` | 缓存参考 mel (推理用) |
| `temp/temp_0502/output_models/ref_S_ori.pt` | 缓存参考语义 (推理用) |
| `temp/temp_0502/inference_results/comparison/` | 三家对比音频 |

### 2.1 总览：数据流

```
推理:
  source.wav → Whisper → S_alt → LengthRegulator → cond
  target.wav → Whisper → S_ori → LengthRegulator → prompt
    │                                CAMPPlus → style
    │                                RMVPE → F0_alt/F0_ori
    │                target_mel
    │                    │
    │    cond + prompt + style + mel + F0
    │                    │
    └────────────────────┼──→ CFM.inference → mel → BigVGAN → 输出.wav
```

训练时有额外一步：花丸音频先走 OpenVoice Timbre Shifter 做音色扰动，再走 Whisper 提取语义 token，形成 S_alt（内容=花丸、音色=随机人）。这迫使 CFM 学会从"任意音色的内容 token"恢复花丸的 mel。

### 2.2 核心推理参数

| 参数 | 作用 | 唱歌推荐 |
|------|------|:--:|
| `diffusion_steps` | Euler 积分步数 | 20-50 |
| `f0_condition` | 注入 F0（唱歌必开） | True |
| `inference_cfg_rate` | CFG 强度（风格遵循） | 0.7 |
| `auto_f0_adjust` | 自动对齐音高范围 | False |
| `semi_tone_shift` | 手动半音移调 | 0 |
| `length_adjust` | 生成窗口缩放 | 1.0 |

### 2.3 各模块一览

| 模块 | 输入 | 输出 | 作用 |
|------|------|------|------|
| Whisper-small | 16kHz 波形 (B, T_16k) | 语义 token (B, T_sem, 768) | 提取语音内容（理论说话人无关） |
| RMVPE | 16kHz 波形 (B, T_16k) | F0 (B, T_f0) 每帧 Hz | 提取旋律/基频 |
| Length Regulator | 语义 token + F0 + 长度 | cond (B, T_mel, 768) | 时域对齐 + F0 注入 + VQ 量化 |
| CAMPPlus | 16kHz 波形 → 80band fbank | style (B, 192) | 全局音色编码 |
| CFM / DiT | 噪声 + cond + style + prompt mel | 速度场预测 | 扩散模型，逐步还原 mel |
| BigVGAN | mel (B, 128, T_mel) | 波形 @ 44100Hz | 声码器 |
| OpenVoice ToneColorConverter | 音频 + 源se + 目标se | 音色转换后音频 | 训练时音色扰动 |

---

## 3. V1 训练流程逐步拆解

**文件**：`seed-vc/train.py`，类 `Trainer.train_one_step()`

### Step 1 — Timbre Shifter（音色扰动）
```
花丸音频 → extract_se(花丸) → 得到 speaker_emb (花丸)
se_db 中随机抽一个别人的 speaker_emb
ToneColorConverter.convert(花丸音频, 花丸_se, 别人_se) → perturbed_wav
```
效果：花丸的"内容 + 别人的音色"。每次迭代随机换一个人（从 100004 人中抽）。

### Step 2 — Whisper（语义提取）
```
S_ori = whisper.encoder(waves_16k)          # 花丸原始语义
S_alt = whisper.encoder(perturbed_wav_16k)  # 扰动后语义
```

### Step 3 — RMVPE（F0）
```
F0_ori = rmvpe(waves_16k)  # 花丸的旋律
```

### Step 4 — Length Regulator
```
cond_ori = length_reg(S_ori, target_len=mel_len, f0=F0_ori)
cond_alt = length_reg(S_alt, target_len=mel_len, f0=F0_ori)
```

### Step 5 — 随机 Prompt 长度
```
prompt_len = randint(0, mel_len-1)
cond = cond_alt 复制一份
cond[:prompt_len] = cond_ori[:prompt_len]   # prompt 区用原始语义
```

### Step 6 — CAMPPlus（全局风格）
```
style = campplus(花丸原始音频)  # (B, 192)
```

### Step 7 — CFM 训练
```
loss = cfm(mel（ground truth）, mel_len, prompt_len, cond, style)
```
内部：随机扩散时间 t → 噪声 z → 直线插值 x_t → DiT 预测速度场 v_pred → L1(v_pred, v_target)

### Step 8 — 反向传播
```
loss_total = cfm_loss + commitment_loss*0.05 + codebook_loss*0.15
backward → clip_grad → optimizer.step
```

### 训练 vs 推理对比

| | 训练 | 推理 |
|------|------|------|
| source 语义 token | 花丸+随机音色扰动 | 任意人的 Whisper 输出 |
| prompt 语义 | 原始花丸（10% 概率长度为0） | target 花丸参考音频 |
| mel | 花丸真实 mel（ground truth） | 从噪声逐步还原 |
| style | 花丸的 CAMPPlus | target 花丸的 CAMPPlus |
| CFM 行为 | 学速度场（L1 loss） | Euler 积分还原 mel |

---

## 4. Seed-VC V2 架构与 AR 失败分析

### 4.1 V2 架构

```
V2 推理链路 (--convert-style true):
source音频 → Whisper → HuBERT → BSQ(窄, codebook=32) → narrow token
target音频 → Whisper → HuBERT → BSQ(窄, codebook=32) → narrow token
                                                              ↓
narrow token → AR(自回归) → wide token (codebook=2048) → CFM → BigVGAN → 语音
                                                              ↑
target音频 → Whisper → HuBERT → BSQ(宽, codebook=2048) ──→ prompt
                                 → CAMPPlus ──→ style
                                 → mel_fn ──→ prompt mel
```

### 4.2 AR 模型失败根因

**问题**：`--convert-style true`（启用 AR）时输出为乱码/鬼叫。

**链式定位**：

1. **AR 在只见过花丸窄 token 的情况下微调了 6000 步**。训练数据只有花丸一个人的窄 token（Whisper→HuBERT→BSQ 量化）。
2. **推理时输入 natori_sana 的窄 token**，AR 从未见过这种 token 分布（不同人的 phonetic/prosodic pattern 有差异）。
3. **AR 是自回归 Transformer**，对输入分布敏感。窄 token 在实践中有残余说话人信息泄漏（Whisper+HuBERT 的 disentanglement 不完美）。
4. **一步预测错误，误差累积** → 整个宽 token 序列崩坏 → CFM 照它生成 mel → 鬼叫。

### 4.3 已尝试的调试（均失败）

| 尝试 | 结果 |
|------|------|
| temperature=0.3→0.1, top_p=0.5→0.3 | 仍鬼叫（问题在输入分布，不在采样随机性） |
| AR 早期 checkpoint (step=500) | 仍鬼叫（从一开始就过拟合了花丸窄 token） |
| 微调 CFM + HF 默认 AR（混搭） | CUDA crash（vocab size 不匹配） |

### 4.4 结论

**AR 微调只适用于同一说话人的不同风格转换**（如花丸说话→花丸不同情感）。跨说话人场景（natori→花丸）下，微调后的 AR 不可用。

更根本地说，这个架构**不能实现"任意说话内容→另一个人的说话方式"的迁移**——AR 只做 token 级别的风格染色，不能改变语义内容。花丸的口癖/节奏/换气模式是和语义绑定的，窄→宽 token 映射无法从 natori 的窄 token 中还原这些特征。

---

## 5. V2 推理中 target 参考音频的作用

即便 AR 崩了，target 在 V2 里仍有三处生效：

| target 生效位置 | 形式 | 维度 | 作用 |
|------|------|:--:|------|
| AR 窄 token 上下文 | `tgt_narrow_reduced` 拼在 AR 输入前缀 | (1, T, —) | 告诉 AR 生成的目标风格 |
| AR 宽 token 模板 | `target_content_indices` 作 `prompt_target` | (1, T, —) | AR 的输出格式参照 |
| CFM prompt mel | `target_mel` 固定扩散前 T 帧 | (1, 80, T) | 扩散的初始条件 |
| CFM style | CAMPPlus 192维 | (1, 192) | 每层 DiT 的 AdaLN 调制 |
| CFM prompt_condition | LengthRegulator 输出 | (1, T, 512) | 条件序列前缀 |

---

## 6. YingMusic-SVC vs V1 逐模块对比

### 6.1 Length Regulator

| | V1 | YingMusic |
|------|:---:|:---:|
| 代码量 | 141 行 | 238 行 |
| 核心新增 | — | `use_style_residual` MLP |

YingMusic 多了两个子网络：
```python
self.f0_to_style_proj = nn.Linear(768, 192)    # F0 投影到风格空间
self.f02style_mlp = Sequential(                 # 逐帧风格残差生成器
    Linear(384→192), Mish, Linear(192→192)
)
```

forward 时：F0_emb 投影到 192 维，与全局 style 拼接 → MLP → tanh → 残差 (B, T_mel, 192)。幅度被 `alpha * RMS(全局style)` 限制，仅在有声帧生效。

### 6.2 Flow Matching（CFM）

| | V1 | YingMusic |
|------|:---:|:---:|
| 代码量 | 167 行 | 667 行 |
| 新增 | — | `style_r` 通道 + 能量均衡 loss |

**差异 A**：`style_r` 参数从 LengthRegulator 透传到 DiT estimator。

**差异 B**：能量均衡损失 (`balance_loss=True`)。对 mel 的 128 个频段按内容能量 + 噪声水平生成自适应权重 `w_bc`，高频弱能量段被提权。训练目标是 `L1(pred*sqrt(w), target*sqrt(w))`，而不是均匀 L1。

### 6.3 DiT (Diffusion Transformer)

| | V1 | YingMusic |
|------|:---:|:---:|
| 代码量 | 537 行 | 564 行 |
| 差异 | — | 多了 `style_r` 通道 |

### 6.4 独有辅助模块

| 模块 | 路径 | 作用 |
|------|------|------|
| `f0_normalization.py` | YingMusic 专用 | WORLD vocoder 提取 → 归一化 F0 → 合成 |
| `f0_fix.py` | YingMusic 专用 | F0 扰动：jitter/glide/jump |

### 6.5 Timbre Shifter

| | V1 | YingMusic |
|------|------|------|
| 模型 | OpenVoice ToneColorConverter | RVC-based（120人训） |
| 权重 | `se_db.pt` (100004人) | 未公开 |
| 本地 | ✅ | ❌ |

### 6.6 总结：YingMusic 比 V1 多什么

```
V1 底盘:
  ├── CFM (DiT + 标准 flow matching L1)
  ├── Length Regulator (Wavenet + VQ + F0 embed)
  └── Timbre Shifter (OpenVoice, 100004人)

YingMusic 在 V1 上增量:
  ├── Length Regulator 多了 ──→ use_style_residual MLP (F0-aware timbre adaptor)
  ├── CFM 多了 ──→ style_r 通道 (逐帧风格残差注入)
  │            ──→ energy balance loss (高频加权，平衡 mel 频率)
  ├── Timbre Shifter 换成 ──→ RVC-based (120个歌手，未公开)
  └── 辅助工具 ──→ f0_fix.py, f0_normalization.py
```

---

## 7. RVC Timbre Shifter 现状与替代方案

### 7.1 确认的事实

- YingMusic 的 RVC timbre shifter 为训练时专用（音色扰动），**不在推理 checkpoint 内**
- `YingMusic-SVC-full.pt` 只包含 `cfm` + `length_regulator`，无 shifter 权重
- 官方未公开 RVC shifter 的预训练权重
- 该 shifter 在 120 个 singer 上训练，针对唱歌场景优化

### 7.2 替代方案对比

| 方案 | 跨说话人 | 花丸相似度 | 工作量 |
|------|:---:|:---:|:--:|
| 用 OpenVoice 顶替 RVC | ✅ | 稍弱 | ✅ 已实现（`temp/temp_0502/`） |
| 跳过 shifter | ❌ | 强 | 小 |
| 自己训 RVC shifter | ✅ | ✅ | 巨大 |

### 7.3 结论：已用 OpenVoice 替代

- `se_db.pt`（100004 人 embedding）已在本地
- 自写训练代码中已实现 OpenVoice ToneColorConverter + se_db 随机采样替代 RVC shifter
- 100004 人的覆盖度 >> 120 人，跨说话人泛化更好
- 代价是 singing-specific 质量稍降（高频/颤音保留不如 RVC 专训版）
- Timbre shifter 不在模型内部，替换不影响 checkpoint 加载
- 训练代码路径：`temp/temp_0502/train_yingmusic_ft_spkemb.py`（当前主力版本）

---

## 14. spkemb 60k 失败根因（2026-05-04）

### 14.1 关键数字

```
hanamaru_avg_style.pt (5603条均值) vs 花丸-平-voice.mp3 CAMPPlus:
  cos = 0.724498     (偏差 28%)
  L2  = 8.4301       (avg norm=8.6 vs single norm=12.2)

spkemb 60k spk_embedding vs avg_style:
  cos = 1.000000     (训练从头到尾没动过)
  L2  = 0.0000
```

### 14.2 权重偏移对比

| 模型 | lr | 步数 | CFM cos | 音色 |
|------|:--:|:--:|:--:|------|
| V1 15000步 | 5e-6 | 15000 | 0.9996 | ✅ |
| exp1 | 5e-6 | 12000 | 0.9997 | ✅ |
| cosine | 1e-4 | 3000 | 0.999999 | ✅ (太早停) |
| spkemb 40k | 1e-4 | 40000 | 0.921 | ❌ 太重 |
| spkemb 60k | 1e-4 | 60000 | ~0.90 | ❌ 费丹 |

### 14.3 三维原因叠加

1. **初始化错误**：avg_style 被 MSST 环境噪声污染
2. **spk_embedding 梯度为零**：无法通过训练修正初始化
3. **CFM 在错误 style 下过度训练**：在错误方向上走了 60k 步

### 14.4 验证实验

用 spkemb 60k CFM + `my_inference.py`（实时 CAMPPlus）推理 → 音色恢复"还行"，但声音"厚重"（能量均衡 loss 副作用）。证明：
- 音色差 → avg_style 的锅
- 厚重感 → CFM 过训的锅

### 14.5 mel2 实验

| mel2 来源 | 效果 |
|-----------|------|
| 花丸-平-voice.mp3 | ✅ 正常 |
| 283条训练数据平均 | ❌ 很差 |
| 全零向量 | ❌ 很差 |

mel2 不能简单平均（抹掉时间动态），不能为零（失去扩散锚点）。单条高质量参考是最优解。

---

## 15. v3 实验方案（2026-05-04 ✅ 完成）

### 目标
用单条 CAMPPlus 初始化 spk_embedding，低 lr 控制 CFM 偏移，产出音色好 + 自然的模型。

### 参数

| 参数 | 值 |
|------|:--:|
| 数据集 | 5603条 / 36.4h |
| 起点 | YingMusic-SVC-full.pt |
| init | **hanamaru_prompt_style.pt** (单条 CAMPPlus) |
| base_lr | 3e-5 |
| schedule | warmup 3k → cosine 20k |
| lr_min | 1e-6 |
| 总步数 | 20000 |
| 实际时间 | ~5 小时 |

### 权重偏移（训练全程）

| 步数 | CFM cos | LR cos | spk cos | 阶段 |
|:--:|:--:|:--:|:--:|------|
| 5000 | 0.998109 | 0.961228 | 1.000000 | 早期有哑音 |
| 10000 | 0.996313 | 0.960366 | 1.000000 | 哑音显著减少 |
| 15000 | 0.995953 | 0.959579 | 1.000000 | 基本收敛 |
| 20000 | 0.995964 | 0.959710 | 1.000000 | 15k→20k 几乎无变化 |

### 与 spkemb 60k 对比

| | spkemb 60k | v3 |
|------|:--:|:--:|
| spk init | avg_style ❌ | prompt_style ✅ |
| base_lr | 1e-4 | **3e-5** |
| 总步数 | 60000 | **20000** |
| CFM cos 终值 | ~0.90 | **0.996** |
| 推理方式 | spkemb/CAMPPlus实时 | **CAMPPlus实时 (最佳)** |
| 音色 | 厚重 | **最准** |

### 推理结论

- **v3 20k + my_inference.py (CAMPPlus实时)** = 最佳推理方案，音色最准
- **v3 20k + inference_spkemb.py (spk_embedding查表)** = 效果不如实时 CAMPPlus（即使 spk cos=1.000）
- 早期 5k checkpoint 有明显哑音（style_r 未收敛），15k 已完全消失

### 输出

| 步数 | 文件 |
|:--:|------|
| 5000 | `spkemb_v3_lr3e-5_20k/DiT_epoch_*_step_05000.pth` |
| 10000 | `spkemb_v3_lr3e-5_20k/DiT_epoch_*_step_10000.pth` |
| 15000 | `spkemb_v3_lr3e-5_20k/DiT_epoch_*_step_15000.pth` |
| 20000 | `spkemb_v3_lr3e-5_20k/ft_model.pth` |

### 新增文件

| 文件 | 作用 |
|------|------|
| `_train_spkemb_v2.py` | 支持环境变量注入 lr/init 的训练脚本 |
| `run_spkemb_v2_experiments.py` | 批量实验启动器 |
| `hanamaru_prompt_style.pt` | 单条 CAMPPlus (norm=12.2) |
| `batch_matrix_infer.py` | 矩阵推理脚本（含进度条） |
| `inference_spkemb_zero_mel2.py` | 零 mel2 对比实验 |
| `inference_spkemb_avg_mel2.py` | 平均 mel2 对比实验 |

---

## 16. 四模型矩阵对比最终结论（2026-05-04）

### 16.1 实验设计

8首歌 × 5模型（cosine/exp1/spkemb 60k/v3 20k campplus + V1 15000参照），输出文件见 `矩阵对比/output/`。

### 16.2 核心发现

**清新 ←→ 厚重 是一条连续谱：**

```
cosine(3k)  exp1(12k)  v3(20k)  spkemb(60k)
  最清新      清新柔和    居中      最厚重
  CFM不动     CFM不动    CFM微动    CFM大改
  cos=1.0    cos=0.9997 cos=0.996 cos=0.90
```

- **cosine/exp1 清新** = CFM 几乎不动 + lr低/步数少，保留了预训练模型的干净质感
- **v3 音色最准** = 正确 CAMPPlus init + 适度训练，CFM cos=0.996 在安全区
- **60k 厚重** = 能量均衡 loss 过度训练 CFM mel 生成路径，但 CAMPPlus 实时推理救回了音色方向

### 16.3 实战选型

| 场景 | 推荐 | 推理参数 |
|------|------|------|
| 最像花丸 | V1 15000步 | diff=20, f0=True |
| YingMusic 音色最准 | v3 20k campplus | diff=100, f0=True |
| YingMusic 清新首选 | cosine 3000 | diff=50, f0=True |
| YingMusic 平衡 | exp1 12000 | diff=50, f0=True |
| 厚重风格 | spkemb 60k | diff=50, f0=True |

**所有 YingMusic 模型**统一用 `my_inference.py` + `--target 花丸-平-voice.mp3`（CAMPPlus 实时）。`inference_spkemb.py`（无target）效果不如实时。

### 16.4 训练方法论总结

| 法则 | 内容 |
|------|------|
| **初始化定律** | spk_embedding 梯度≈0，初始化质量 = 最终质量 |
| **CFM 安全区** | cos > 0.95：清新；cos ~0.90：厚重但仍可用（CAMPPlus实时救场） |
| **清新 vs 准确** | 不可兼得。CFM 不动则清新但可能音色不准，CFM 改动则准确但变厚重 |
| **数据量悖论** | 1153条可产出最清新效果（cosine/exp1）；5603条+大lr 反而厚重 |
| **最佳推理** | CAMPPlus 实时 > spk_embedding 查表（即使 cos=1.000） |

---

## 8. 微调 YingMusic 的执行路线

### 8.1 目标

将 YingMusic-SVC-full.pt（CFM + LengthRegulator 含 style_residual）在花丸数据上微调，获得跨说话人能力的专属模型。

### 8.2 已完成事项

1. ✅ **训练脚本** `train_yingmusic_ft_spkemb.py`，基于 V1 的 `train.py` 改写
2. ✅ **改 config 路径** → `YingMusic-SVC.yml`（`use_style_residual` 参数）
3. ✅ **改模型构建** → YingMusic 的 `build_model()` 正确构造带 style_residual 的 LengthRegulator
4. ✅ **加载 YingMusic checkpoint** → `YingMusic-SVC-full.pt` 的 `cfm` + `length_regulator` 权重
5. ✅ **OpenVoice Timbre Shifter** → 从 V1 `train.py` 复制 ToneColorConverter + se_db 随机采样
6. ✅ **开启能量均衡 loss** → `balance_loss=True`（YingMusic CFM 已实现）
7. ✅ **多轮训练完成** → spkemb 60k (费丹→存档), v3 20k (✅主力), cosine 3000 (✅最清新), exp1 12000 (✅清新柔和)

### 8.3 预计效果

- CFM 用 YingMusic 预训练 + 能量均衡 loss → 唱歌表现力提升
- Length Regulator 用 style_residual MLP → 动态音色调制
- OpenVoice shifter 保持跨说话人泛化
- 花丸数据微调 + 可学习 spk_embedding → 音色向花丸靠拢

### 8.4 相关文件清单

| 文件 | 位置 |
|------|------|
| V1 训练脚本 | `seed-vc/train.py` |
| V1 微调脚本 (lr=5e-6) | `seed-vc/train_lr_5e-6.py` |
| V2 训练脚本 | `seed-vc/train_v2.py` |
| V1 推理脚本 | `seed-vc/inference.py` |
| V2 推理脚本 | `seed-vc/inference_v2.py` |
| YingMusic 推理脚本 | `YingMusic-SVC/my_inference.py` |
| YingMusic 模型配置 | `YingMusic-SVC/configs/YingMusic-SVC.yml` |
| YingMusic checkpoint | `YingMusic-SVC/YingMusic-SVC-full.pt` |
| YingMusic length_regulator | `YingMusic-SVC/modules/length_regulator.py` |
| YingMusic flow_matching | `YingMusic-SVC/modules/flow_matching.py` |
| OpenVoice se_db | `seed-vc/modules/openvoice/checkpoints_v2/converter/se_db.pt` |
| V1 DiT | `seed-vc/modules/diffusion_transformer.py` |
| YingMusic DiT | `YingMusic-SVC/modules/diffusion_transformer.py` |
| **自写训练脚本** | **`temp/temp_0502/train_yingmusic_ft_spkemb.py`** |
| **自写推理脚本** | **`temp/temp_0502/inference_spkemb.py`** |

---

## 9. 权重偏移分析（V1 vs YingMusic）

### 9.1 分析方法

使用 `compare_weights.py` / `batch_compare.py` / `compare_v1.py`，将微调 checkpoint 与各自预训练基座逐参数计算 cosine similarity 和 L2 距离。

### 9.2 V1 权重偏移（基座：Seed-VC DiT 预训练）

| 实验 | 步数 | CFM cos | LR cos | 特点 |
|------|:--:|:--:|:--:|------|
| my_run (R1) | 1750 | 0.9998 | 0.916 | 48条，LR一次性对齐 |
| hanamaru_full_fav (R2) | 6000 | **0.9996** | **0.916** | 1153条，CFM 几乎不动 |
| hanamaru_full_fav_step15000 | 15000 | **0.9996** | **0.916** | ★最佳，续训无进一步变化 |

**结论**：V1 的 CFM 在整个训练过程中仅偏移了 0.0004 cosine —— 基本是预训练原样。LR 在最初 500 步完成对齐调整后冻结在 cos=0.916。这证明「CFM 轻触 + F0 引导」策略极其高效。

### 9.3 YingMusic 权重偏移（基座：YingMusic-SVC-full.pt）

| 步数 | CFM cos | CFM ΔL2 | LR cos | LR ΔL2 |
|:--:|:--:|:--:|:--:|:--:|
| 5000 | 0.980 | 497 | 0.942 | 22 |
| 10000 | 0.958 | 763 | 0.924 | 37 |
| 20000 | 0.934 | 1017 | 0.896 | 53 |
| 30000 | 0.924 | 1126 | 0.881 | 60 |
| **40000** | **0.921** | **1155** | **0.876** | **62** |
| 预计 60000 | ~0.918 | — | ~0.873 | — |

### 9.4 对比解读

| | V1 | YingMusic |
|------|------|------|
| CFM 偏移量 | 0.0004 (几乎不动) | 0.079 (大幅改写) |
| LR 偏移量 | 0.084 (一次性跳变) | 0.124 (持续学习) |
| 学习模式 | 轻触微调 | 深度适应 |
| 优势 | 旋律稳定性 ⭐⭐⭐⭐⭐ | 自然度更高（能量均衡 + style_r） |
| 代价 | 高频刺耳 | CFM 改写幅度大，需更多步数收敛 |

V1 证明了只动 LR 不动 CFM 已经够好；YingMusic 额外引入能量均衡 loss 和逐帧 style_residual，CFM 必须配合改写才能发挥。试听已确认 YingMusic 的自然度优于 V1。
