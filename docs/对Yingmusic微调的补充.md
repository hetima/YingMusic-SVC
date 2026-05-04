> ⚠️ **最新更新（2026-05-04）**：YingMusic 四模型（60k spkemb / exp1 12000 / cosine 3000 / v3 20k campplus）在 8 首歌曲上完成矩阵对比。**各有优劣**：v3 20k campplus 音色最准，cosine 和 exp1 声音更清新，60k spkemb 虽费丹但有独特厚重感。
>
> ⚠️ **历史更新（2026-05）**：原作者（GiantAILab）至今**仍未开源训练代码**，官方仓库仍是 inference-only。但我们已经**自己写好了训练脚本**，经过三轮迭代。

**YingMusic-SVC** 的**社区方案和具体教程**目前相对有限（2025 年底发布的项目），官方以 **inference-only** 为主，**训练/微调代码尚未开源**（官方 GitHub 仓库只有推理脚本、accompany separation、Gradio app，没有 `train.py` 或完整训练 pipeline）。

### 1. 官方仓库与社区现状
- **官方仓库**：https://github.com/GiantAILab/YingMusic-SVC （MIT License，基于 Seed-VC 扩展）。
  - 包含：`accom_separation/`（伴奏分离）、`gradio_app.py`（GUI）、`my_inference.py` / `my_infer.sh`（CLI 推理）、`configs/`、预训练 checkpoint（通过 HF 或脚本下载）。
  - **无训练代码**：README 明确只有 inference 相关。论文描述了三阶段训练（CPT → Robust SFT → Flow-GRPO RL），但未提供可直接运行的脚本。
  - Forks：少量（如 juanjosehr14/YingMusic-SVC），主要是镜像或轻微包装，无新增训练支持。
  - Discussions/Issues：GitHub 和 HF 页面上讨论较少（stars ~130，forks ~13），目前没有活跃的社区训练分享或 Discord 服务器。
  - HF 页面：https://huggingface.co/GiantAILab/YingMusic-SVC （model card 简短，主要指向 GitHub，无额外训练示例）。

- **相关项目**：
  - **YingMusic-Singer**（同团队）：零样本歌声合成（SVS），有 inference 代码，但非 SVC。
  - **Seed-VC**：YingMusic-SVC 的基础，其微调代码（`train.py`）和 OpenVoice 模块是自写训练脚本的关键依赖。
  - 社区讨论散见于 Reddit (r/so_vits_svc)、中文论坛或论文评述（如 themoonlight.io），但多为 inference 使用或论文解读，无成熟的 YingMusic-SVC 微调社区方案。

- **论文参考**：arXiv:2512.04793。论文描述了三阶段训练（CPT → Robust SFT → Flow-GRPO RL），但必须注意论文中的 RVC timbre shifter 需要额外训练数据（120 个歌手），官方未提供该模块权重。**自写训练代码时已将其替换为 Seed-VC 的 OpenVoice se_db 扰动方案**（见下方第 3 节）。

**结论**：原作者确实没开源训练代码，但我们已经基于 Seed-VC 架构 + YingMusic 的 CFM/length_regulator 模块，从零写出了可用的训练脚本。以下第 3 节记录已实现方案。

### 2. 具体使用教程（Inference 部分，已部署可直接用）
你的 venv（py310 + cu128）已就绪，进入 `E:\AIscene\AISVCs\YingMusic-SVC` 并激活 venv 后：

#### **模式**
- **伴奏分离（Accompany Separation）**：处理真实歌曲（带和声/伴奏），输出干净 lead vocal（提升后续 SVC 质量）。基于 Band RoFormer。
- **Zero-shot SVC**：源唱歌音频（可带伴奏或已分离） → 目标歌手参考音频 → 输出保留 melody + lyrics + 目标 timbre 的转换歌声。强在真实世界鲁棒性（harmony 干扰、F0 误差）。
- **全 pipeline**：分离 → SVC →（可选）与伴奏混合 + BigVGAN vocoder。
- **Gradio GUI**：可视化操作，支持上传 source/target，调节参数。

#### **输入要求**
- **Source**：唱歌音频（wav/flac/mp3，推荐 44.1kHz，长度 5-60s 最佳；可带伴奏或干净 vocal）。
- **Target Reference**：目标歌手短音频（1-30s，干净说话/唱歌均可，影响 timbre 相似度）。
- **可选**：semi-tone shift（跨性别常用 +12），伴奏混合权重 γ。

#### **可调节参数**（主要继承/扩展 Seed-VC + singing-specific）
- Diffusion steps / sampling steps（推理质量 vs 速度，推荐 10-50；12GB VRAM 建议低值防 OOM）。
- CFG rate / inference cfg（控制相似度 vs 多样性）。
- F0 condition / pitch shift（半音移调）。
- Timbre shifter strength（RVC-based，影响 disentanglement）。
- Energy-balanced loss 相关（推理中不直接调，但影响高频细节）。
- GUI 中：上传文件后，通常有 sliders for steps、cfg、shift 等（具体看 `gradio_app.py` 源码）。

#### **步骤**
1. **伴奏分离**（推荐先做，提升鲁棒性）：
   ```cmd
   cd YingMusic-SVC/accom_separation
   bash infer.sh   # 或 Windows 等价：python infer.py --input your_song.wav --output separated/
   ```
   - 会下载/使用 Band RoFormer 模型，输出 lead vocal。

2. **CLI SVC 推理**：
   ```cmd
   cd YingMusic-SVC
   bash my_infer.sh
   ```
   - 编辑 `my_infer.sh` 或 `my_inference.py` 调整路径、参数（source、target、output、steps、cfg 等）。
   - 示例命令（参考源码）：
     ```cmd
     python my_inference.py \
       --source separated_lead.wav \
       --target reference_singer.wav \
       --output outputs/result.wav \
       --diffusion_steps 25 \
       --cfg_rate 0.7 \
       --semi_tone_shift 0
     ```

3. **Gradio GUI**（最方便）：
   ```cmd
   python gradio_app.py
   ```
   - 浏览器访问 http://127.0.0.1:7860。
   - 操作：上传 source（可带伴奏或已分离）、target reference，选择是否分离、调节 sliders，生成。

**VRAM 优化（你的 12GB）**：
- 用 fp16 / torch.compile（若支持）。
- 短音频 + 低 diffusion steps（10-20 for real-time-ish）。
- 监控 `nvidia-smi`；若 OOM，降低 batch（推理通常单样本）或 steps。
- BigVGAN vocoder 可能吃显存，必要时分步跑。

**输出**：转换后的唱歌音频（.wav），melody/lyrics 保留好，timbre 接近 target，尤其在 noisy/harmony 场景。

参考：官方 README、`gradio_app.py` / `my_inference.py` 源码、arXiv 论文 inference pipeline 部分。

### 3. 自写训练方案（已实现，原作者未开源）

> **原作者（GiantAILab）至今未开源训练代码**。以下训练脚本全部由我们自己编写，位于本仓库根目录下。

#### 训练脚本清单

| 文件 | 用途 | 关键参数 |
|------|------|----------|
| `train_yingmusic_ft.py` | 基础版 | lr=5e-6，CAMPPlus 实时提取 speaker embedding |
| `train_yingmusic_ft_cosine.py` | Cosine 版 | lr=1e-4 warmup → cosine decay |
| `train_yingmusic_ft_spkemb.py` | **SpkEmb 版（当前主力）** | 可学习 `spk_embedding`（192维），warmup=3000, lr=1e-4→1e-6 |

#### 核心架构改动：RVC Timbre Shifter → OpenVoice se_db 扰动

论文中的 RVC timbre shifter 模块需要额外在 120 个歌手上预训练，官方未提供该权重。我们的替代方案：

```
原文：音频 → RVC Timbre Shifter（不可用）→ perturbed 语义 → CFM
我们：音频 → OpenVoice ToneColorConverter + se_db（100004人）→ perturbed 语义 → CFM
```

具体流程（以 `train_one_step` 为核心）：

```
1. 提取 speaker embedding (se_batch) ← OpenVoice ToneColorConverter
2. 从 se_db 随机采样一个他人 embedding (ref_se)   ← 替代 RVC 扰动
3. convert(waves, se_batch → ref_se)               ← 生成音色扰动版音频
4. Whisper/XLSR 提取语义：S_ori（原始）+ S_alt（扰动后）
5. RMVPE 提取 F0
6. CAMPPlus（或 spk_embedding）提取 style vector y
7. Length Regulator: S_alt + y → alt_cond + style_r
8. CFM Flow Matching: 随机 prompt_len 混合 ori/alt cond，balance_loss=True
9. 更新 cfm + length_regulator + spk_embedding
```

**与论文的主要差异**：

| 模块 | 论文方案 | 我们的实现 |
|------|----------|-----------|
| Timbre Shifter | RVC-based（120歌手预训练） | OpenVoice se_db 随机采样（100004人） |
| Speaker Embedding | CAMPPlus（实时） | 基础版用 CAMPPlus；spkemb 版用可学习 Embedding |
| F0-aware Adaptor | 论文特有 | 通过 `length_regulator(style=y, return_style_residual=True)` 的 style_r 间接实现 |
| Energy-balanced Loss | λ=0.4 | `balance_loss=True`（YingMusic CFM 原生支持） |
| Harmony Augmentation | 混入伴奏 | 未实现（依赖预处理分离质量） |
| Flow-GRPO RL | 三阶段 RL | 未实现（12GB 显存不够 + reward model 复杂） |

#### 启动训练

```powershell
cd YingMusic-SVC

# SpkEmb 版（当前使用）
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

输出保存在 `output_models/<run_name>/`，checkpoint 命名格式：`DiT_epoch_XXXXX_step_XXXXX.pth`，最终模型 `ft_model.pth`。

#### 训练历史

| 实验 | 脚本 | 步数 | lr | 数据量 | 结果 |
|------|------|:--:|:--:|------|------|
| yingmusic_exp1 | ft.py | 12000 | 5e-6 | 1153条 | ✅ **YingMusic 主力**（CAMPPlus实时, 最平衡） |
| yingmusic_cosine | ft_cosine.py | 3000 | 1e-4 warmup | 1153条 | ✅ 单听有进步 |
| **yingmusic_spkemb_60k** | **ft_spkemb.py** | **60000** | **1e-4 cosine** | **5603条 / 36.4h** | **❌ 费丹→存档（有独特厚重感）** |
| **spkemb_v3_lr3e-5_20k** | **_train_spkemb_v2.py** | **20000** | **3e-5 cosine** | **5603条 / 36.4h** | **✅ v3 主力 (warmup 3k)** |

#### 训练过程中的重要发现（2026-05）

**ft_dataset 长音频截断修复**：原始代码对 >30s 的音频直接跳过，浪费大量数据。已改为截取前 30 秒，不再丢弃超长音频。

**调度器断点续训 Bug 修复**：`load_checkpoint(load_only_params=True)` 在续训时丢掉了 optimizer 和 scheduler 状态，导致断电重启后 lr 从 0 重新 warmup。已修复为：首次加载预训练权重时 `load_only=True`，续训自己 checkpoint 时 `load_only=False`。

**Loss 曲线解释**：
| 区间 | Loss | 原因 |
|------|:--:|------|
| 0~500 步 | 0.75→0.58 快速下降 | lr 从 0 爬升，模型快速适应 |
| 500~3000 步 | 0.58→0.72 上升 | warmup 继续推高 lr 至 1e-4，梯度噪声放大 |
| 3000~43000 步 | 0.72→0.48 持续下降 | cosine decay 生效，收敛进入正道 |
| 预计 60000 步 | ~0.40~0.45 | 接近收敛终点 |

#### 22 个模型权重偏移全量分析

使用 `batch_compare.py` 对比了全部 22 个 checkpoint 相对 `YingMusic-SVC-full.pt` 的偏移：

| 实验 | 步数范围 | CFM cos 终值 | LR cos 终值 | 结论 |
|------|------|:--:|:--:|------|
| yingmusic_exp1 | 500→12000 | **0.9997** | 0.964 | ❌ lr=5e-6 太低，权重几乎没变 |
| yingmusic_cosine | 1000→3000 | **0.999999** | 0.965 | ❌ 太早停止，warmup 刚结束 |
| **yingmusic_spkemb_60k** | 5000→40000 | **0.921** | **0.876** | ✅ 持续学习，收敛趋势明确 |

spkemb 的收敛曲线：
```
5000  .980  ████████████████████████████████████████████
10000  .958  ████████████████████████████████████████
15000  .944  ████████████████████████████████████
20000  .934  ██████████████████████████████
25000  .927  █████████████████████████
30000  .924  ███████████████████
35000  .922  ██████████████
40000  .921  ██████████  ← cos 下降速度趋近于 0，收敛中
```

#### 与 Seed-VC V1 的权重偏移对比

| 模型 | CFM cos | LR cos | 特点 |
|------|:--:|:--:|------|
| V1 (★最佳唱歌模型) | **0.9996** | 0.916 | CFM 几乎不动，LR 一次性对齐后冻结 |
| YingMusic 40000步 | **0.921** | 0.876 | CFM 大幅改写，LR 持续学 style_residual |

V1 证明了「CFM 轻触 + F0 引导」就能产出最佳唱歌效果。YingMusic 额外改动了 20 倍的参数量，但这是能量均衡 loss 和 style_residual 机制所需的——试听已确认 YingMusic 的自然度优于 V1（柔和度、高频细节更好）。

#### F0→style 侧枝冻结分析

`f0_to_style_proj` 和 `f02style_mlp` 共 6 个参数在 40000 步后 cos=1.000 —— 完全没学到。根因是 gradient 经过 `tanh → RMS normalize → clamp(max=1.0)` 三重截断后几乎为零。不影响效果：F0 信息已通过 `f0_embedding → x = x + f0_emb` 主通路充分进入 CFM。

### 4. 推荐下一步

- **v3 20k campplus**（lr=3e-5, 100 diff steps）→ **音色最准**，YingMusic 唱歌主力。
- **cosine 3000** → 声音**清新**（8首歌对比中与exp1并列最清新），lr=1e-4但步数短，CFM cos=0.999999。
- **exp1 12000**（lr=5e-6）→ 音色**清新柔和**，CFM cos=0.9997，与cosine同属清新派。
- **spkemb 60k**（lr=1e-4, CAMPPlus实时推理）→ 音色**厚重**，CFM cos≈0.90，适合需要厚重感的歌曲。
- 替代方案：Seed-VC V1 15000 步作为稳定 baseline（音色最像，但偏刺耳）。
- 可继续探索：降低 lr 至 1e-5 微调 cosine 模型，寻找 "清新感" 的极限。


---

## 5. spkemb 60k 失败根因分析（2026-05-04）

### 5.1 问题表现

spkemb 60k 所有 checkpoint（5k/10k/15k/.../60k）输出音色均严重偏离花丸。最早 checkpoint（5k步）效果最差。

### 5.2 根因

**`hanamaru_avg_style.pt` 与真实花丸音色偏离过大。**

```
hanamaru_avg_style.pt  ← 5603条 MSST 分离产物的 CAMPPlus 简单平均
花丸-平-voice.mp3 CAMPPlus  ← 单条干净、高质量参考

cos = 0.724498     (差了 28% 方向)
L2  = 8.4301       (avg norm=8.6 vs single norm=12.2, 幅度缩水 30%)
```

5603 条训练数据来自 272 首歌的 MSST 分离产物，每首的残留混响、去噪 artifact、环境差异被 CAMPPlus 捕捉后求平均——**录音环境噪声也被平均进了音色**，导致 192 维向量偏离花丸真实音色方向。

### 5.3 spk_embedding 梯度为零

训练 60k 步后：

```
hanamaru_avg_style.pt vs ft_model.pth 中 spk_embedding:
  cos = 1.000000    （完全没变）
  L2  = 0.0000
```

`spk_embedding` 参与了训练但梯度几乎为 0——初始化就是终点。问题从一开始就锁死了。

### 5.4 CFM 被过度训练

| 模型 | CFM cos vs 预训练 | 音色 |
|------|:--:|------|
| V1 15000步 | 0.9996 | ✅ 音色好 |
| exp1 12000步 | 0.9997 | ✅ 音色好 |
| spkemb 40000步 | 0.921 | ❌ 太重 |
| spkemb 60000步 | ~0.90 | ❌ 整炉费丹 |

CFM 在错误 style 条件下学到大量补偿性变换，lr=1e-4 加速了这一过程。

### 5.5 关键验证：CAMPPlus 实时 vs spk_embedding

用 spkemb 60k 的 CFM 权重 + `my_inference.py`（实时 CAMPPlus 从 `花丸-平-voice.mp3` 提取）推理：

→ 音色恢复 "还行"，但声音"厚重"

**结论**：60k CFM 本身还能用，但被能量均衡 loss 过度训练导致了厚重感。音色差完全是 avg_style 的锅。

### 5.6 mel2 对比实验

| 来源 | 效果 |
|------|------|
| `花丸-平-voice.mp3` 单条 | ✅ 正常 |
| 283条训练数据平均 | ❌ 很差 |
| 全零向量 | ❌ 很差 |

mel2 作为扩散锚点，不能简单平均化——平均 mel 抹掉了时间动态（attack/decay/颤音包络），零向量则失去锚定导致输出劣化。当前方案（单条高质量音频）是最优解。

### 5.7 教训

- **CAMPPlus 对录音环境极其敏感**，多源数据简单平均会引入噪声
- **YingMusic 微调必须 `CFM cos > 0.95`**（lr≤3e-5 或步数≤20k），否则音色损失不可逆
- **spk_embedding 梯度极弱**，必须用高质量单条 CAMPPlus 初始化，不能指望训练修正

---

## 6. 四模型矩阵对比（8首歌曲 · 2026-05-04）

### 6.1 对比模型

| 模型 | 训练步数 | lr | 数据量 | CFM cos | 推理方式 | 推理步数 |
|------|:--:|:--:|------|:--:|------|:--:|
| **spkemb 60k** | 60000 | 1e-4 | 5603条 | ~0.90 | my_inference.py (CAMPPlus实时) | 50 |
| **exp1** | 12000 | 5e-6 | 1153条 | 0.9997 | my_inference.py (CAMPPlus实时) | 50 |
| **cosine** | 3000 | 1e-4 | 1153条 | 0.999999 | my_inference.py (CAMPPlus实时) | 50 |
| **v3 20k campplus** | 20000 | 3e-5 | 5603条 | 0.9960 | my_inference.py (CAMPPlus实时) | 100 |
| **V1 15000** (参照) | 15000 | 5e-6 | 1153条 | 0.9996 | inference.py (f0_cond=True) | 70 |

测试歌曲：8首花丸翻唱干声（SummerMemory、星空、Talking、恋愛サーキュレーション 等），BVID 见 `矩阵对比/output/`。

### 6.2 综合评价（按 "清新感" 排序）

| 模型 | 音色像度 | 清新感 | 厚重感 | 自然度 | 定位 |
|------|:--:|:--:|:--:|:--:|------|
| **cosine 3000** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐ | 轻快歌曲首选 |
| **exp1 12000** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | 清新柔和派 |
| **v3 20k campplus** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 音色准度标杆 |
| **spkemb 60k** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 特殊厚重风格 |
| V1 15000 (参照) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | 音色最像但刺耳 |

### 6.3 核心发现：训练深度 vs 音色取向

```
清新 ←——————————————————————————————————————————→ 厚重/准确

cosine(3k)   exp1(12k)        v3(20k)          spkemb(60k)
↑ lr=1e-4     ↑ lr=5e-6       ↑ lr=3e-5        ↑ lr=1e-4
  CFM≈不动       CFM≈不动       CFM微动(0.004)     CFM大幅改写(0.10)
  数据1153       数据1153       数据5603          数据5603
```

**规律**：

1. **清新感来自 CFM 几乎不动**：cosine（cos=0.999999）和 exp1（cos=0.9997）的 CFM 保留了预训练模型的干净质感，lr低 + 步数少 = 只学了 style_r，未触动 mel 生成核心。

2. **厚重感来自 CFM 被改写**：spkemb 60k（cos≈0.90）的能量均衡 loss 过度训练了 CFM mel 生成路径，导致输出"厚重"。v3 20k（cos=0.996）居中——比 cosine/exp1 厚重但比 60k 轻得多。

3. **音色准度来自正确的 style + 适度训练**：v3 20k campplus 用单条高质量 CAMPPlus 初始化 + real-time CAMPPlus 推理，音色最准。60k 的 CFM 虽然被过度训练，但实时 CAMPPlus 救回了音色方向。

4. **数据量不是关键**：cosine/exp1 只用 1153 条，效果反而清新。v3/60k 用 5603 条，但过大 lr/步数导致厚重。

### 6.4 实战选型建议

| 歌曲类型 | 推荐模型 | 理由 |
|------|------|------|
| 轻快/可爱系 | **cosine 3000** 或 **exp1** | 声音清新、不厚重 |
| 抒情慢歌 | **v3 20k campplus** | 音色最准、情感表达好 |
| 需要厚重/力量感 | **spkemb 60k** | 独特的厚重音色 |
| 音色优先（不管风格） | **v3 20k campplus** | 最像花丸 |
| 平衡首选 | **exp1 12000** | 清新 + 音色都不错 |

### 6.5 与 V1 的定位关系

- **V1 15000** 仍是音色最像的模型，但偏刺耳（V1 的均匀 L1 loss 对高频约束不足）
- **YingMusic 四个模型**统一比 V1 柔和（能量均衡 loss 的功劳），代价是音色像度略逊
- 两套模型**互补**：V1 追求音色极致，YingMusic 追求听感舒适