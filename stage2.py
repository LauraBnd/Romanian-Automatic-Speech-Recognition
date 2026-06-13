from datasets import load_dataset
from pathlib import Path
import soundfile as sf
import numpy as np
import csv
import json
import gc
import subprocess
import io
import random

OUTPUT_DIR = Path("stage2_opensource")
VOXPOPULI_DIR = OUTPUT_DIR / "voxpopuli_ro" / "train"
CV_DIR = OUTPUT_DIR / "common_voice_ro" / "train"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
CSV_PATH = OUTPUT_DIR / "manifest.csv"
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint.json"

CV_CORPUS_DIR = Path("Mozilla/cv-corpus-25.0-2026-03-09/ro")
CV_CLIPS_DIR = CV_CORPUS_DIR / "clips"
CV_TSV = CV_CORPUS_DIR / "validated.tsv"

VOXPOPULI_MAX_HOURS = 3.0
CV_MAX_HOURS = 3.0

for d in [VOXPOPULI_DIR, CV_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"voxpopuli_done": False, "cv_done": False, "manifest": [], "hours": 0.0}

def save_checkpoint(state):
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def process_voxpopuli(state):
    manifest = state["manifest"]
    total_seconds = state["hours"] * 3600
    max_seconds = VOXPOPULI_MAX_HOURS * 3600

    print("Loading VoxPopuli Romanian...")
    vp = load_dataset("facebook/voxpopuli", "ro")
    vp_train = vp["train"].shuffle(seed=42)

    for i, sample in enumerate(vp_train):
        if total_seconds >= max_seconds:
            print(f"  [voxpopuli] 5h cap reached at sample {i}.")
            break

        audio = sample["audio"]
        array = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]
        duration = len(array) / sr

        filepath = VOXPOPULI_DIR / f"voxpopuli_{i:06d}.wav"
        sf.write(filepath, array, sr)

        manifest.append({
            "file": str(filepath),
            "transcript": sample.get("normalized_text") or "",
            "duration_seconds": round(duration, 3),
            "sample_rate": sr,
            "source": "voxpopuli_train",
        })
        total_seconds += duration

        if i % 200 == 0:
            gc.collect()
            state["hours"] = round(total_seconds / 3600, 4)
            state["manifest"] = manifest
            save_checkpoint(state)
            print(f"  [voxpopuli] {i} files | {state['hours']:.3f}h")

    state["hours"] = round(total_seconds / 3600, 4)
    state["manifest"] = manifest
    state["voxpopuli_done"] = True
    save_checkpoint(state)
    print(f"VoxPopuli done. {state['hours']:.3f}h | {len(manifest)} samples\n")

def mp3_to_array(mp3_path: Path):
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(mp3_path),
         "-ar", "16000", "-ac", "1", "-f", "wav", "pipe:1"],
        capture_output=True,
    )
    if result.returncode != 0:
        return None, None
    data, sr = sf.read(io.BytesIO(result.stdout))
    return data.astype(np.float32), sr

def process_common_voice(state):
    manifest = state["manifest"]
    cv_seconds = 0.0

    if not CV_TSV.exists():
        print(f"  [cv] validated.tsv not found at {CV_TSV}. Skipping.")
        state["cv_done"] = True
        save_checkpoint(state)
        return

    print(f"Loading Common Voice 25.0 Romanian (full) from {CV_CORPUS_DIR}...")

    with open(CV_TSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    rng = random.Random(42)
    rng.shuffle(rows)

    skipped = 0
    for i, row in enumerate(rows):
        if cv_seconds >= CV_MAX_HOURS * 3600:
            print(f"  [cv] 3h cap reached at row {i}.")
            break

        clip_name = row.get("path", "")
        mp3_path = CV_CLIPS_DIR / clip_name
        if not mp3_path.exists():
            skipped += 1
            continue

        array, sr = mp3_to_array(mp3_path)
        if array is None:
            skipped += 1
            continue

        duration = len(array) / sr
        filepath = CV_DIR / f"cv25_{i:06d}.wav"
        sf.write(filepath, array, sr)

        manifest.append({
            "file": str(filepath),
            "transcript": row.get("sentence", "").strip(),
            "duration_seconds": round(duration, 3),
            "sample_rate": sr,
            "source": "common_voice_25",
        })
        cv_seconds += duration

        if i % 200 == 0:
            gc.collect()
            state["hours"] = round((state["hours"] * 3600 + cv_seconds) / 3600, 4) if i == 0 else round(
                sum(e["duration_seconds"] for e in manifest) / 3600, 4
            )
            state["manifest"] = manifest
            save_checkpoint(state)
            print(f"  [cv] {i} rows | cv_hours={cv_seconds/3600:.3f} | skipped={skipped}")

    state["hours"] = round(sum(e["duration_seconds"] for e in manifest) / 3600, 4)
    state["manifest"] = manifest
    state["cv_done"] = True
    save_checkpoint(state)
    print(f"Common Voice done. cv_hours={cv_seconds/3600:.3f}h | skipped={skipped}\n")

state = load_checkpoint()

if not state.get("voxpopuli_done"):
    process_voxpopuli(state)
else:
    print(f"VoxPopuli already done ({state['hours']:.3f}h). Skipping.")

if not state.get("cv_done"):
    process_common_voice(state)
else:
    print("Common Voice already done. Skipping.")

with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
    json.dump(state["manifest"], f, ensure_ascii=False, indent=2)

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=["file", "transcript", "duration_seconds", "sample_rate", "source"]
    )
    writer.writeheader()
    writer.writerows(state["manifest"])

by_source = {}
for entry in state["manifest"]:
    s = entry["source"]
    by_source.setdefault(s, {"count": 0, "hours": 0.0})
    by_source[s]["count"] += 1
    by_source[s]["hours"] += entry["duration_seconds"] / 3600

print(f"Total hours   : {state['hours']:.3f}h")
print(f"Total samples : {len(state['manifest'])}")
for src, stats in by_source.items():
    print(f"  {src:35s} {stats['count']:5d} clips  {stats['hours']:.3f}h")