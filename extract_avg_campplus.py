import os, sys, torch, glob, time, soundfile as sf
os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
import torchaudio, torchaudio.compliance.kaldi as kaldi

device = torch.device('cuda:0')
data_dir = 'train_data'
output_path = 'output_models/hanamaru_avg_style.pt'

def load_campplus():
    from modules.campplus.DTDNN import CAMPPlus
    from hf_utils import load_custom_model_from_hf
    model = CAMPPlus(feat_dim=80, embedding_size=192)
    ckpt_path = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", config_filename=None)
    sd = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(sd)
    model.eval()
    model.to(device)
    return model

campplus = load_campplus()
print("CAMPPlus loaded")

audio_files = []
for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a']:
    audio_files.extend(glob.glob(os.path.join(data_dir, 'speaker1', ext)))

print(f"Found {len(audio_files)} audio files")
if len(audio_files) == 0:
    print("ERROR: No audio files found!")
    sys.exit(1)

styles = []
for i, fp in enumerate(audio_files):
    try:
        wav, sr = sf.read(fp)
        if len(wav.shape) > 1: wav = wav.mean(axis=1)
        wav_t = torch.tensor(wav).unsqueeze(0).float()
        if sr != 16000:
            wav_t = torchaudio.transforms.Resample(sr, 16000)(wav_t)
        wav_t = wav_t[:, :16000 * 25].to(device)

        feat = kaldi.fbank(wav_t, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)

        with torch.no_grad():
            style = campplus(feat.unsqueeze(0))
        styles.append(style.squeeze(0).cpu())

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(audio_files)}")
    except Exception as e:
        print(f"  skip {fp}: {e}")

styles = torch.stack(styles)
avg_style = styles.mean(dim=0)
print(f"\nAverage style over {len(styles)} files")
print(f"  shape: {avg_style.shape}")
print(f"  norm:  {avg_style.norm().item():.4f}")
print(f"  std:   {styles.std(dim=0).mean().item():.6f}")

os.makedirs(os.path.dirname(output_path), exist_ok=True)
torch.save(avg_style, output_path)
print(f"\nSaved to: {output_path}")
