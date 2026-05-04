import subprocess, os, re, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

INPUT_DIR  = r'E:\AIscene\AISVCs\temp\temp_0502\msst_temp\03_final'
OUTPUT_DIR = r'E:\AIscene\AISVCs\temp\temp_0502\train_data\speaker1-plus'
QUIET_THRESH = "-40"
MIN_SILENCE = "0.15"
WORKERS = 6
CREATE_FLAGS = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
print_lock = Lock()

def log(msg):
    with print_lock: print(msg, flush=True)

def run_ff(args):
    return subprocess.run(args, capture_output=True, text=True, encoding='utf-8', errors='replace', creationflags=CREATE_FLAGS)

def detect_silence(path):
    proc = run_ff(['ffmpeg','-i',path,'-af',f'silencedetect=n={QUIET_THRESH}dB:d={MIN_SILENCE}','-f','null','-'])
    silences = []
    cs = None
    for line in proc.stderr.splitlines():
        m = re.search(r'silence_start:\s*([0-9.e+\-]+)', line)
        if m: cs = float(m.group(1))
        m = re.search(r'silence_end:\s*([0-9.e+\-]+)\s*\|\s*silence_duration:\s*([0-9.e+\-]+)', line)
        if m and cs is not None:
            silences.append((cs, float(m.group(1)), float(m.group(2))))
            cs = None
    return silences

def get_duration(path):
    proc = run_ff(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',path])
    return float(proc.stdout.strip())

def decide_cuts(silences, duration, basename):
    if duration <= 30: return [(0, duration, f'{basename}.wav')]
    segs = []; T = 0.0; i = 0
    while True:
        rem = duration - T
        if rem <= 30: segs.append((T, duration, f'{basename}_seg{i:03d}.wav')); break
        cut = False
        for ws, we in [(T+15,T+30),(T+10,T+30),(T+5,T+30)]:
            cands = [(s,e,d) for s,e,d in silences if (ws<=e<=we) or (ws<=s<=we)]
            if not cands: continue
            cands.sort(key=lambda x: x[0])
            t1 = [(s,e,d) for s,e,d in cands if d>=2.0 and s>=T+10]
            if t1: s,e,_=t1[-1]; segs.append((T,s,f'{basename}_seg{i:03d}.wav')); T=s; i+=1; cut=True; break
            t2 = [(s,e,d) for s,e,d in cands if e>=T+22 and d>=0.5]
            if t2: s,e,_=t2[-1]; segs.append((T,e,f'{basename}_seg{i:03d}.wav')); T=e; i+=1; cut=True; break
            t3 = [(s,e,d) for s,e,d in cands if T+15<=e<T+22 and d>=1.0]
            if t3: s,e,_=t3[-1]; segs.append((T,e,f'{basename}_seg{i:03d}.wav')); T=e; i+=1; cut=True; break
            if cands: s,e,_=cands[-1]; segs.append((T,e,f'{basename}_seg{i:03d}.wav')); T=e; i+=1; cut=True; break
        if not cut: segs.append((T,T+29.5,f'{basename}_seg{i:03d}_hard.wav')); T+=29.5; i+=1
    return segs

def strip_silence(in_path, out_path):
    """ffmpeg trim leading/trailing silence + compress long internal gaps."""
    cmd = [
        'ffmpeg', '-y', '-i', in_path,
        '-af', 'silenceremove=start_threshold=-35dB:start_duration=0.3:stop_threshold=-35dB:stop_duration=0.3:detection=peak',
        out_path
    ]
    run_ff(cmd)
    return os.path.exists(out_path) and os.path.getsize(out_path) > 1000

COUNT = 0

def process_song(f):
    global COUNT
    full = os.path.join(INPUT_DIR, f)
    basename = os.path.splitext(f)[0]
    try:
        silences = detect_silence(full)
        duration = get_duration(full)
        segments = decide_cuts(silences, duration, basename)

        for start, end, seg_name in segments:
            out_path = os.path.join(OUTPUT_DIR, seg_name)
            if os.path.exists(out_path):
                continue
            # Cut segment
            tmp = out_path + '.cut.wav'
            run_ff(['ffmpeg','-y','-i', full, '-ss', str(start), '-t', str(end-start), '-c', 'copy', tmp])
            # Strip silence from segment
            ok = strip_silence(tmp, out_path)
            if os.path.exists(tmp): os.remove(tmp)
            if not ok:
                continue
            # Check duration
            d = get_duration(out_path)
            if d < 1.0:
                os.remove(out_path)
                continue
            COUNT += 1

        return ('OK', f, len(segments))
    except Exception as e:
        log(f'[ERROR] {f}: {e}')
        return ('ERROR', f, 0)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Only process _Vocals_dry_dry.wav, skip _other
    files = sorted([f for f in os.listdir(INPUT_DIR)
                    if f.lower().endswith('_vocals_dry_dry.wav')])
    log(f'Processing {len(files)} dry_dry files...')
    ok, errs = 0, 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(process_song, f): f for f in files}
        for future in as_completed(futures):
            status, fname, segs = future.result()
            if status == 'OK': ok += 1; log(f'[OK] {fname} -> {segs} segs')
            else: errs += 1
    log(f'DONE: OK={ok} Errors={errs} TotalSegments={COUNT}')

if __name__ == '__main__': main()
