import os, sys, argparse
os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
import torch, yaml, librosa, torchaudio, soundfile as sf, time
import numpy as np
from modules.commons import recursive_munch, build_model, load_checkpoint
from hf_utils import load_custom_model_from_hf

FIXED_REF = r"E:\AIscene\AISVCs\source_voices\花丸-平-voice.mp3"
CACHE_DIR = "output_models"

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def extract_and_cache(ref_path, cache_dir):
    """Pre-extract mel2 and S_ori from reference audio. Cache to disk."""
    cache_mel = os.path.join(cache_dir, "ref_mel2.pt")
    cache_sori = os.path.join(cache_dir, "ref_S_ori.pt")

    if os.path.exists(cache_mel) and os.path.exists(cache_sori):
        mel2 = torch.load(cache_mel, map_location='cpu')
        S_ori = torch.load(cache_sori, map_location='cpu')
        print(f"Loaded cached mel2 {list(mel2.shape)} and S_ori {list(S_ori.shape)}")
        return mel2, S_ori

    print(f"Extracting reference features from {ref_path} ...")
    from modules.audio import mel_spectrogram
    from transformers import AutoFeatureExtractor, WhisperModel

    ref_audio, sr = librosa.load(ref_path, sr=44100)
    ref_audio_t = torch.tensor(ref_audio).unsqueeze(0).float()[: sr * 25]
    ref_16k_t = torch.tensor(librosa.resample(ref_audio, orig_sr=sr, target_sr=16000)).unsqueeze(0)

    mel_fn_args = {
        "n_fft": 2048, "win_size": 2048, "hop_size": 512,
        "num_mels": 128, "sampling_rate": sr,
        "fmin": 0, "fmax": None, "center": False,
    }
    mel2 = mel_spectrogram(ref_audio_t, **mel_fn_args)

    whisper_model = WhisperModel.from_pretrained("openai/whisper-small", torch_dtype=torch.float16)
    del whisper_model.decoder
    whisper_fe = AutoFeatureExtractor.from_pretrained("openai/whisper-small")
    inputs = whisper_fe([ref_16k_t.squeeze(0).numpy()], return_tensors="pt", return_attention_mask=True, sampling_rate=16000)
    with torch.no_grad():
        out = whisper_model.encoder(
            whisper_model._mask_input_features(inputs.input_features, attention_mask=inputs.attention_mask).to(torch.float16),
            return_dict=True,
        )
    S_ori = out.last_hidden_state.float()[:, : ref_16k_t.size(-1) // 320 + 1]

    torch.save(mel2, cache_mel)
    torch.save(S_ori, cache_sori)
    print(f"Cached mel2 {list(mel2.shape)} and S_ori {list(S_ori.shape)}")
    return mel2, S_ori


def load_models(ckpt_path, config_path):
    config = yaml.safe_load(open(config_path, "r"))
    model_params = recursive_munch(config["model_params"])
    model_params.dit_type = 'DiT'

    from modules.flow_matching import CFM
    from modules.length_regulator import InterpolateRegulator

    model = build_model(model_params, stage="DiT")
    model, _, _, _ = load_checkpoint(model, None, ckpt_path, load_only_params=True, ignore_modules=[], is_distributed=False)
    for key in model:
        model[key].eval()
        model[key].to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    state = torch.load(ckpt_path, map_location='cpu')
    if 'spk_embedding' in state:
        spk_emb = torch.nn.Embedding(1, 192)
        spk_emb.load_state_dict(state['spk_embedding'])
        spk_emb.to(device)
        spk_emb.eval()
        print("Loaded spk_embedding from checkpoint")
    else:
        print("WARNING: spk_embedding not found in checkpoint, falling back to zeros")
        spk_emb = torch.nn.Embedding(1, 192).to(device)
        spk_emb.eval()

    vocoder_type = model_params.vocoder.type
    if vocoder_type == 'bigvgan':
        from modules.bigvgan import bigvgan
        vocoder = bigvgan.BigVGAN.from_pretrained(model_params.vocoder.name, use_cuda_kernel=False)
        vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(device)
    else:
        raise ValueError(f"Unknown vocoder: {vocoder_type}")

    from modules.rmvpe import RMVPE
    rmvpe_path = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None)
    f0_extractor = RMVPE(rmvpe_path, is_half=False, device=device)

    return model, spk_emb, vocoder, f0_extractor, config


def crossfade(chunk1, chunk2, overlap):
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    if len(chunk2) < overlap:
        chunk2[:overlap] = chunk2[:overlap] * fade_in[:len(chunk2)] + (chunk1[-overlap:] * fade_out)[:len(chunk2)]
    else:
        chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2


@torch.no_grad()
def inference(source_path, output_path, model, spk_emb, vocoder, f0_extractor, config,
              diffusion_steps=50, inference_cfg_rate=0.7):
    sr = config['preprocess_params']['sr']
    hop_length = config['preprocess_params']['spect_params']['hop_length']

    from modules.audio import mel_spectrogram
    from transformers import AutoFeatureExtractor, WhisperModel

    whisper_model = WhisperModel.from_pretrained("openai/whisper-small", torch_dtype=torch.float16).to(device)
    del whisper_model.decoder
    whisper_fe = AutoFeatureExtractor.from_pretrained("openai/whisper-small")

    mel_fn_args = {
        "n_fft": config['preprocess_params']['spect_params']['n_fft'],
        "win_size": config['preprocess_params']['spect_params']['win_length'],
        "hop_size": config['preprocess_params']['spect_params']['hop_length'],
        "num_mels": config['preprocess_params']['spect_params']['n_mels'],
        "sampling_rate": sr, "fmin": 0, "fmax": None, "center": False,
    }

    source_audio = librosa.load(source_path, sr=sr)[0]
    source_t = torch.tensor(source_audio).unsqueeze(0).float().to(device)
    source_16k = torch.tensor(librosa.resample(source_audio, orig_sr=sr, target_sr=16000)).unsqueeze(0).to(device)

    if source_16k.size(-1) <= 16000 * 30:
        inputs = whisper_fe([source_16k.squeeze(0).cpu().numpy()], return_tensors="pt", return_attention_mask=True, sampling_rate=16000)
        with torch.no_grad():
            out = whisper_model.encoder(
                whisper_model._mask_input_features(inputs.input_features, attention_mask=inputs.attention_mask).to(device).half(),
                return_dict=True,
            )
        S_alt = out.last_hidden_state.float()[:, : source_16k.size(-1) // 320 + 1].to(device)
    else:
        overlapping_time = 5
        S_alt_list = []
        buffer = None
        traversed_time = 0
        while traversed_time < source_16k.size(-1):
            if buffer is None:
                chunk = source_16k[:, traversed_time:traversed_time + 16000 * 30]
            else:
                chunk = torch.cat(
                    [buffer, source_16k[:, traversed_time:traversed_time + 16000 * (30 - overlapping_time)]],
                    dim=-1)
            inputs = whisper_fe([chunk.squeeze(0).cpu().numpy()], return_tensors="pt", return_attention_mask=True, sampling_rate=16000)
            with torch.no_grad():
                out = whisper_model.encoder(
                    whisper_model._mask_input_features(inputs.input_features, attention_mask=inputs.attention_mask).to(device).half(),
                    return_dict=True,
                )
            S_chunk = out.last_hidden_state.float()[:, : chunk.size(-1) // 320 + 1].to(device)
            if traversed_time == 0:
                S_alt_list.append(S_chunk)
            else:
                S_alt_list.append(S_chunk[:, 50 * overlapping_time:])
            buffer = chunk[:, -16000 * overlapping_time:]
            traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
        S_alt = torch.cat(S_alt_list, dim=1)

    mel2, S_ori = extract_and_cache(FIXED_REF, CACHE_DIR)
    mel2 = mel2.to(device)

    style = spk_emb(torch.zeros(1, dtype=torch.long, device=device))

    source_mel = mel_spectrogram(source_t, **mel_fn_args)
    source_mel_len = source_mel.size(2)
    target_mel_len = mel2.size(2)

    F0_ori = f0_extractor.infer_from_audio(
        torch.tensor(librosa.resample(librosa.load(FIXED_REF, sr=44100)[0][:44100*25], orig_sr=44100, target_sr=16000)).to(device),
        thred=0.03
    )
    F0_alt = f0_extractor.infer_from_audio(source_16k[0], thred=0.03)
    F0_ori = torch.from_numpy(F0_ori).to(device)[None]
    F0_alt_t = torch.from_numpy(F0_alt).to(device)[None]

    voiced_alt = F0_alt_t[F0_alt_t > 1]
    voiced_ori = F0_ori[F0_ori > 1]
    log_f0_alt = torch.log(F0_alt_t + 1e-5)
    shifted = log_f0_alt.clone()
    if len(voiced_ori) > 0 and len(voiced_alt) > 0:
        med_ori = torch.median(torch.log(voiced_ori + 1e-5))
        med_alt = torch.median(voiced_alt)
        shifted[F0_alt_t > 1] = log_f0_alt[F0_alt_t > 1] - torch.log(med_alt + 1e-5) + med_ori
    shifted_f0 = torch.exp(shifted)

    prompt_condition, _, _, _, _, style_prompt = model.length_regulator(
        S_ori.to(device), ylens=torch.LongTensor([target_mel_len]).to(device),
        f0=F0_ori, style=style, return_style_residual=True)
    cond, _, _, _, _, style_cond = model.length_regulator(
        S_alt, ylens=torch.LongTensor([source_mel_len]).to(device),
        f0=shifted_f0, style=style, return_style_residual=True)

    max_context_window = sr // hop_length * 30
    max_source_window = max_context_window - target_mel_len
    overlap_frame_len = 16
    overlap_wave_len = overlap_frame_len * hop_length
    processed_frames = 0
    generated_wave_chunks = []
    previous_chunk = None

    while processed_frames < cond.size(1):
        chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]
        is_last_chunk = processed_frames + max_source_window >= cond.size(1)
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)

        if style_cond is not None:
            chunk_style_cond = style_cond[:, processed_frames:processed_frames + max_source_window]
            cat_style_cond = torch.cat([style_prompt, chunk_style_cond], dim=1)
        else:
            cat_style_cond = None

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            vc_target = model.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(device),
                mel2, style, None, diffusion_steps,
                inference_cfg_rate=inference_cfg_rate,
                style_r=cat_style_cond,
            )
            vc_target = vc_target[:, :, target_mel_len:]

        vc_wave = vocoder(vc_target.float()).squeeze()
        vc_wave = vc_wave[None, :]
        if processed_frames == 0:
            if is_last_chunk:
                generated_wave_chunks.append(vc_wave[0].cpu().numpy())
                break
            generated_wave_chunks.append(vc_wave[0, :-overlap_wave_len].cpu().numpy())
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
        elif is_last_chunk:
            output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len)
            generated_wave_chunks.append(output_wave)
            processed_frames += vc_target.size(2) - overlap_frame_len
            break
        else:
            output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(), overlap_wave_len)
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len

    full_audio = np.concatenate(generated_wave_chunks)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, full_audio.T, sr)
    print(f"Saved: {output_path} ({len(full_audio)/sr:.1f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Path to source audio")
    parser.add_argument("--output", type=str, default="./inference_results/spkemb_output.wav")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to spkemb checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to YingMusic-SVC.yml")
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--inference-cfg-rate", type=float, default=0.7)
    args = parser.parse_args()

    model, spk_emb, vocoder, f0_ext, config = load_models(args.checkpoint, args.config)
    inference(args.source, args.output, model, spk_emb, vocoder, f0_ext, config,
              args.diffusion_steps, args.inference_cfg_rate)
