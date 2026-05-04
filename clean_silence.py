import os, subprocess, shutil, tempfile, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import soundfile as sf
import numpy as np

BASE = r'E:\AIscene\AISVCs\temp\temp_0502\train_data'
SR = 44100
MIN_GAP = 5.0    # silences longer than this get compressed
TARGET_GAP = 2.0 # compressed to this
SILENCE_THRESH = -35  # dB
print_lock = Lock()

def log(msg):
    with print_lock: print(msg, flush=True)


def get_all_files():
    """Collect all wav files from speaker1 and speaker1-plus."""
    files = []
    for d in ['speaker1', 'speaker1-plus']:
        sd = os.path.join(BASE, d)
        if os.path.isdir(sd):
            for f in os.listdir(sd):
                if f.lower().endswith(('.wav', '.mp3', '.flac')):
                    files.append(os.path.join(sd, f))
    return files


def trim_silence(in_path, out_path):
    """Trim leading/trailing silence using ffmpeg silenceremove."""
    cmd = [
        'ffmpeg', '-y', '-i', in_path,
        '-af', f'silenceremove=start_threshold={SILENCE_THRESH}dB:start_duration=0.3:stop_threshold={SILENCE_THRESH}dB:stop_duration=0.3:detection=peak',
        out_path
    ]
    subprocess.run(cmd, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0)
    return os.path.exists(out_path) and os.path.getsize(out_path) > 1000


def detect_silences(path):
    """Return list of (start, end, duration) for silent segments."""
    cmd = [
        'ffmpeg', '-i', path,
        '-af', f'silencedetect=n={SILENCE_THRESH}dB:d=0.3',
        '-f', 'null', '-'
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0)
    silences = []
    current_start = None
    for line in r.stderr.splitlines():
        m = re.search(r'silence_start:\s*([0-9.e+\-]+)', line)
        if m:
            current_start = float(m.group(1))
        m = re.search(r'silence_end:\s*([0-9.e+\-]+)\s*\|\s*silence_duration:\s*([0-9.e+\-]+)', line)
        if m and current_start is not None:
            end = float(m.group(1))
            dur = float(m.group(2))
            silences.append((current_start, end, dur))
            current_start = None
    return silences


def get_duration(path):
    r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', path],
                       capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0)
    return float(r.stdout.strip()) if r.stdout.strip() else 0


def build_trimmed_with_gaps(in_path, out_path, long_silences):
    """Rebuild audio: keep non-silent segments, replace long silences with TARGET_GAP seconds of silence."""
    data, sr = sf.read(in_path)
    if data.ndim == 2:
        data = data.mean(axis=1)
    total_samples = len(data)
    duration = total_samples / sr

    long_silences.sort(key=lambda x: x[0])
    keep_segments = []
    cursor = 0.0
    for start, end, dur in long_silences:
        if start > cursor:
            seg_start_sample = int(cursor * sr)
            seg_end_sample = int(start * sr)
            if seg_end_sample > seg_start_sample:
                keep_segments.append(data[seg_start_sample:seg_end_sample])
        keep_segments.append(np.zeros(int(TARGET_GAP * sr), dtype=data.dtype))
        cursor = end
    if cursor < duration:
        keep_segments.append(data[int(cursor * sr):])

    if keep_segments:
        result = np.concatenate(keep_segments)
        sf.write(out_path, result, sr)


def process_one(fp):
    """Process a single file: trim -> detect long gaps -> rebuild if needed."""
    try:
        # Step 1: trim start/end silence
        tmp = fp + '.trim.wav'
        ok = trim_silence(fp, tmp)
        if not ok:
            log(f'  SKIP (trim failed): {os.path.basename(fp)}')
            return 'skip'

        dur = get_duration(tmp)
        if dur < 1.0:
            os.remove(tmp)
            os.remove(fp)
            log(f'  REMOVE (too short): {os.path.basename(fp)}')
            return 'removed'

        # Step 2: detect long internal silences
        silences = detect_silences(tmp)
        long_silences = [(s, e, d) for s, e, d in silences if d > MIN_GAP]

        if not long_silences:
            # Clean — just replace original with trimmed version
            os.remove(fp)
            os.rename(tmp, fp)
            return 'clean'

        # Step 3: rebuild with compressed gaps
        rebuild = fp + '.rebuild.wav'
        build_trimmed_with_gaps(tmp, rebuild, long_silences)
        os.remove(tmp)

        dur2 = get_duration(rebuild)
        if dur2 < 1.0:
            os.remove(rebuild)
            os.remove(fp)
            return 'removed'

        os.remove(fp)
        os.rename(rebuild, fp)
        return 'rebuilt'

    except Exception as e:
        # cleanup
        for p in [fp + '.trim.wav', fp + '.rebuild.wav']:
            if os.path.exists(p): os.remove(p)
        log(f'  ERROR {os.path.basename(fp)}: {e}')
        return 'error'


def main():
    files = get_all_files()
    log(f'Processing {len(files)} files...')
    stats = {'clean': 0, 'rebuilt': 0, 'removed': 0, 'skip': 0, 'error': 0}

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(process_one, f): f for f in files}
        i = 0
        for fut in as_completed(futures):
            status = fut.result()
            stats[status] = stats.get(status, 0) + 1
            i += 1
            if i % 500 == 0:
                log(f'  Progress: {i}/{len(files)}')

    log(f'\nDone: {stats}')

    # Final stats
    remaining = 0; total_sec = 0.0
    for d in ['speaker1', 'speaker1-plus']:
        sd = os.path.join(BASE, d)
        if os.path.isdir(sd):
            remaining += len(os.listdir(sd))
    log(f'Files remaining: {remaining}')


if __name__ == '__main__':
    main()
