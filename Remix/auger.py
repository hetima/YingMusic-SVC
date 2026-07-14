import torch
import torchaudio
from typing import List, Optional, Tuple
import librosa
import numpy as np
import math
import torch.nn.functional as F
import soundfile as sf
from pedalboard import Delay, Reverb
from pathlib import Path

def ensure_2d(wave):
    """确保音频是 [C, T] 格式"""
    if wave.dim() == 1:
        wave = wave.unsqueeze(0)  # [T] -> [1, T]
    return wave

def safe_normalize(wave, peak=0.98):
    """安全归一化，防止削波"""
    max_val = wave.abs().max()
    if max_val > peak:
        wave = wave * (peak / max_val)
    return wave

def align_length(processed_wave, target_length):
    """
    对齐音频长度
    Args:
        processed_wave: 处理后的音频张量 [C, T]
        target_length: 目标长度
    Returns:
        长度对齐后的音频张量
    """
    current_length = processed_wave.shape[1]
    if current_length > target_length:
        return processed_wave[:, :target_length]  # 截断
    elif current_length < target_length:
        padding = target_length - current_length
        return torch.nn.functional.pad(processed_wave, (0, padding))  # 填充零
    else:
        return processed_wave

EFFECT_PARAMS = {
    # Echo 参数 (SoX格式)
    'echo': {
        'delay_ms': 110.0,  # 延迟时间(毫秒)
        'feedback': 0.2,  # 反馈强度 (0-1)
        'wet': 0.18,  # 混合比例 (0-1)
    },

    # Reverb 参数 (SoX格式)
    'reverb': {
        'reverberance': 55,  # 混响强度 (0-100)
        'hf_damping': 65,  # 高频阻尼 (0-100)
        'room_scale': 65,  # 房间大小 (0-100)
        'stereo_depth': 100,  # 立体声深度 (0-100)
        'pre_delay_ms': 25,  # 预延迟(毫秒)
        'wet': 0.25,  # 混合比例 (0-1)
    },

}

@torch.no_grad()
def _apply_pedalboard_effect(wave: torch.Tensor, sr: int, effect):
    """
    确保 pedalboard 在 CPU/float32 上运行，并把结果搬回原 device/dtype。
    wave: [T] 或 [C,T]
    """
    wave2d = ensure_2d(wave)
    ref_device, ref_dtype = wave2d.device, wave2d.dtype
    w_cpu = wave2d.to("cpu", dtype=torch.float32, non_blocking=False).contiguous().numpy()
    y_cpu = effect.process(w_cpu, sample_rate=sr)
    y = torch.from_numpy(np.asarray(y_cpu)).to(device=ref_device, dtype=ref_dtype)
    return y

def apply_echo(wave, sr, delay_ms=150.0, feedback=0.4, wet=0.35):
    wave = ensure_2d(wave)
    original_length = wave.shape[1]
    effect = Delay(
        delay_seconds=max(0.0, float(delay_ms)) / 1000.0,
        feedback=float(np.clip(feedback, 0.0, 1.0)),
        mix=1.0,
    )
    y = _apply_pedalboard_effect(wave, sr, effect)
    y = align_length(y, original_length)
    y = (1 - wet) * wave + wet * y
    return safe_normalize(y)



def apply_reverb(wave, sr, reverberance=80, hf_damping=40, room_scale=120,
                stereo_depth=100, pre_delay_ms=25, wet=0.5):
    wave = ensure_2d(wave)
    original_length = wave.shape[1]
    effect = Reverb(
        room_size=float(np.clip(room_scale, 0.0, 100.0)) / 100.0,
        damping=float(np.clip(hf_damping, 0.0, 100.0)) / 100.0,
        wet_level=1.0,
        dry_level=0.0,
        width=float(np.clip(stereo_depth, 0.0, 100.0)) / 100.0,
    )
    # pedalboard の Reverb には pre-delay がないため、入力を遅らせて近似する。
    pre_delay_samples = max(0, int(float(pre_delay_ms) * sr / 1000.0))
    effect_wave = wave
    if pre_delay_samples:
        effect_wave = F.pad(effect_wave, (pre_delay_samples, 0))
    y = _apply_pedalboard_effect(effect_wave, sr, effect)
    y = align_length(y, original_length)
    y = (1 - wet) * wave + wet * y
    return safe_normalize(y)

@torch.no_grad()
def echo_then_reverb_save(
    audio_path: str,
    out_path: str,
    instrument_path: str | None = None,
    echo_kwargs: dict | None = None,
    reverb_kwargs: dict | None = None,
    target_sr: int | None = None,
    peak: float = 0.98,
):
    """
    读取 audio_path -> 先回声(SoX) -> 再混响(SoX) -> 保存到 out_dir
    - 输出文件名：<basename>.wav（basename 与输入一致）
    - 默认保持原始采样率；可通过 target_sr 指定输出采样率
    - echo_kwargs / reverb_kwargs 会覆盖默认参数（见下方 EFFECT_PARAMS）
    """
    audio_path = Path(audio_path)
    # 1) 读取
    wave, sr = torchaudio.load(str(audio_path))  # [C, T], float32 or int
    wave = wave.to(torch.float32)
    # 统一到 [-1,1]
    if wave.abs().max() > 1.0:
        wave = wave / (wave.abs().max() + 1e-12)

    # 2) 采样率处理（可选）
    out_sr = sr if target_sr is None else int(target_sr)
    if out_sr != sr:
        resampler = torchaudio.transforms.Resample(sr, out_sr)
        wave = resampler(wave)
        sr = out_sr

    # 3) 效果参数合并
    ep = dict(EFFECT_PARAMS)  # 你上面代码里的全局默认
    if echo_kwargs:
        ep["echo"] = {**ep.get("echo", {}), **echo_kwargs}
    if reverb_kwargs:
        ep["reverb"] = {**ep.get("reverb", {}), **reverb_kwargs}

    y=wave
    # 4) 先回声 -> 再混响（都已在你的实现里保证对齐与规范化）
    y = apply_echo(
        y, sr,
        delay_ms=ep["echo"]["delay_ms"],
        feedback=ep["echo"]["feedback"],
        wet=ep["echo"]["wet"]
    )
    y = apply_reverb(
        y, sr,
        reverberance=ep["reverb"]["reverberance"],
        hf_damping=ep["reverb"]["hf_damping"],
        room_scale=ep["reverb"]["room_scale"],
        stereo_depth=ep["reverb"]["stereo_depth"],
        pre_delay_ms=ep["reverb"]["pre_delay_ms"],
        wet=ep["reverb"]["wet"]
    )
    # 5) 最终安全归一化
    y = safe_normalize(y, peak=peak)

    # + accompany
    y_aug = y.cpu().numpy().mean(axis=0)
    instrumental, _ = librosa.load(instrument_path, sr=sr, mono=True)
    instrumental = instrumental / np.max(np.abs(instrumental))
    min_len = min(len(y_aug), len(instrumental))

    # 响度归一化
    y_aug = amplify_to_peak(y_aug,target_db=-2)

    instrumental = amplify_to_peak(instrumental, target_db=-2)

    mix_audio = y_aug[:min_len] + instrumental[:min_len]

    # 保存最终混音结果
    sf.write(out_path, mix_audio, sr)

    return str(out_path)

def amplify_to_peak(wav: np.ndarray, target_db: float = -2.0) -> np.ndarray:
    """
    把音频整体放大/缩小到目标峰值 (例如 -2 dBFS)
    wav: numpy array from librosa.load
    target_db: 目标峰值 (默认 -2 dB)
    """
    target_amp = 10 ** (target_db / 20.0)   # -2 dBFS ≈ 0.794
    peak = np.max(np.abs(wav))
    if peak < 1e-9:
        return wav  # 全静音，不处理
    gain = target_amp / peak
    return wav * gain

