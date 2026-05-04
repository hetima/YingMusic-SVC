import os, sys, argparse, shutil, time, types, glob
import torch
sys.path.insert(0, r"E:\AIscene\AISVCs\YingMusic-SVC")
os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'

AUDIO_DIR = r"E:\AIscene\AISVCs\temp\temp_0502\inference_results\comparison\矩阵对比\歌声音频"
OUTPUT_DIR = r"E:\AIscene\AISVCs\temp\temp_0502\inference_results\comparison\矩阵对比\output"
TARGET_REF = r"E:\AIscene\AISVCs\source_voices\花丸-平-voice.mp3"
CONFIG_PATH = r"E:\AIscene\AISVCs\YingMusic-SVC\configs\YingMusic-SVC.yml"
DIFF_STEPS = 50

MODELS_DIR = r"E:\AIscene\AISVCs\temp\temp_0502\output_models"

MODELS = [
    ("exp1_12000", f"{MODELS_DIR}/yingmusic_exp1/ft_model.pth"),
    ("cosine_3000", f"{MODELS_DIR}/yingmusic_cosine/ft_model.pth"),
    ("spkemb_05000", f"{MODELS_DIR}/yingmusic_spkemb_60k/DiT_epoch_00000_step_05000.pth"),
    ("spkemb_10000", f"{MODELS_DIR}/yingmusic_spkemb_60k/DiT_epoch_00001_step_10000.pth"),
    ("spkemb_15000", f"{MODELS_DIR}/yingmusic_spkemb_60k/DiT_epoch_00002_step_15000.pth"),
    ("spkemb_20000", f"{MODELS_DIR}/yingmusic_spkemb_60k/DiT_epoch_00003_step_20000.pth"),
    ("spkemb_25000", f"{MODELS_DIR}/yingmusic_spkemb_60k/DiT_epoch_00004_step_25000.pth"),
    ("spkemb_30000", f"{MODELS_DIR}/yingmusic_spkemb_60k/DiT_epoch_00005_step_30000.pth"),
    ("spkemb_35000", f"{MODELS_DIR}/yingmusic_spkemb_60k/DiT_epoch_00006_step_35000.pth"),
    ("spkemb_40000", f"{MODELS_DIR}/yingmusic_spkemb_60k/DiT_epoch_00007_step_40000.pth"),
    ("spkemb_50000", f"{MODELS_DIR}/yingmusic_spkemb_60k/DiT_epoch_00001_step_50000.pth"),
    ("spkemb_60000", f"{MODELS_DIR}/yingmusic_spkemb_60k/ft_model.pth"),
]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

audio_files = sorted([
    f for f in os.listdir(AUDIO_DIR)
    if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg'))
])

total_tasks = len(MODELS) * len(audio_files)
current_task = 0
start_time = time.time()

print(f"{'='*60}")
print(f"MODELS: {len(MODELS)}  |  SONGS: {len(audio_files)}  |  TOTAL: {total_tasks} tasks")
print(f"STEPS: {DIFF_STEPS}  |  OUTPUT: {OUTPUT_DIR}")
print(f"{'='*60}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

from my_inference import load_models_api, run_inference

for mi, (model_tag, ckpt_path) in enumerate(MODELS):
    model_start = time.time()
    print(f"\n[MODEL {mi+1}/{len(MODELS)}] {model_tag}  ({ckpt_path})")

    if not os.path.isfile(ckpt_path):
        print(f"  SKIP: checkpoint not found")
        current_task += len(audio_files)
        continue

    args = types.SimpleNamespace()
    args.source = ""
    args.target = TARGET_REF
    args.diffusion_steps = DIFF_STEPS
    args.checkpoint = ckpt_path
    args.expname = "batch"
    args.cuda = device
    args.fp16 = True
    args.config = CONFIG_PATH
    args.length_adjust = 1.0
    args.inference_cfg_rate = 0.7
    args.f0_condition = True
    args.semi_tone_shift = None
    args.output = "./outputs"
    args.accompany = None
    os.makedirs(args.output, exist_ok=True)

    print(f"  Loading model...")
    try:
        model_bundle = load_models_api(args, device=device)
    except Exception as e:
        print(f"  LOAD FAILED: {e}")
        current_task += len(audio_files)
        continue

    for si, audio_name in enumerate(audio_files):
        current_task += 1
        source_path = os.path.join(AUDIO_DIR, audio_name)
        song_tag = os.path.splitext(audio_name)[0]
        output_name = f"{model_tag}_{song_tag}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_name)

        elapsed = time.time() - start_time
        avg_per_task = elapsed / max(current_task - len(audio_files) + si + 1, 1)
        eta_sec = avg_per_task * (total_tasks - (current_task - 1)) if current_task > 1 else 0

        h = int(eta_sec // 3600)
        m = int((eta_sec % 3600) // 60)
        pct = current_task / total_tasks * 100

        print(f"  [{current_task:3d}/{total_tasks} {pct:5.1f}%] [{model_tag}] {audio_name[:30]}...  ETA: {h}h{m:02d}m", flush=True)

        if os.path.exists(output_path):
            print(f"    (skipped, exists)")
            continue

        task_start = time.time()
        try:
            args.source = source_path
            run_inference(args, model_bundle, device=device)

            exp_dir = os.path.join(args.output, args.expname)
            wavs = sorted(glob.glob(os.path.join(exp_dir, "*.wav")), key=os.path.getmtime, reverse=True)
            if wavs:
                shutil.move(wavs[0], output_path)
            else:
                print(f"    WARNING: no output found in {exp_dir}")

            dt = time.time() - task_start
            print(f"    Done in {dt:.0f}s", flush=True)
        except Exception as e:
            dt = time.time() - task_start
            print(f"    FAILED after {dt:.0f}s: {type(e).__name__}: {e}", flush=True)

    dt_model = time.time() - model_start
    print(f"  Model done in {dt_model/60:.1f}m")

total_elapsed = time.time() - start_time
h = int(total_elapsed // 3600)
m = int((total_elapsed % 3600) // 60)
print(f"\n{'='*60}")
print(f"ALL DONE in {h}h{m:02d}m  |  {total_tasks} tasks")
print(f"{'='*60}")
