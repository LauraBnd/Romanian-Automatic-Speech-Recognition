import json
import csv
import random
import subprocess
from pathlib import Path

MANIFEST_PATH = next(Path(".").glob("**/manifest.json"), Path("manifest.json"))
RECORDINGS_DIR = next(Path(".").glob("**/recordings"), Path("recordings"))
OUTPUT_DIR = Path("stage1_dataset")

NOISE_START_ID = "5ea8b70c-2f6f-4f7b-98f5-d6a585b73a54"
WHISPER_START_ID = "f9f806bd-c2cf-4203-9db2-f4026a8f309c"

DIFFICULTY_LABELS = {
    1: "Clear speech, no noise, normal pace",
    2: "Light background noise",
    3: "Whispered speech",
}

RANDOM_SEED = 42

def assign_difficulties(recordings: list) -> dict:
    ids = [r["id"] for r in recordings]
    noise_idx = ids.index(NOISE_START_ID) if NOISE_START_ID in ids else len(ids)
    whisper_idx = ids.index(WHISPER_START_ID) if WHISPER_START_ID in ids else len(ids)

    result = {}
    for i, rec_id in enumerate(ids):
        if i >= whisper_idx:
            result[rec_id] = 3
        elif i >= noise_idx:
            result[rec_id] = 2
        else:
            result[rec_id] = 1
    return result

def stratified_split(recordings: list, difficulty_map: dict):
    groups = {1: [], 2: [], 3: []}
    for rec in recordings:
        groups[difficulty_map[rec["id"]]].append(rec)

    rng = random.Random(RANDOM_SEED)
    for g in groups.values():
        rng.shuffle(g)

    assignment = {}
    for level, group in groups.items():
        n = len(group)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)
        for i, rec in enumerate(group):
            if i < train_end:
                assignment[rec["id"]] = "train"
            elif i < val_end:
                assignment[rec["id"]] = "validation"
            else:
                assignment[rec["id"]] = "test"

    return assignment

def convert_webm_to_wav(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ar", "16000", "-ac", "1", str(dst)],
        capture_output=True,
        text=True
    )
    return result.returncode == 0

def build_dataset():
    print(f"manifest.json : {MANIFEST_PATH.resolve()}")
    print(f"recordings/   : {RECORDINGS_DIR.resolve()}")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    recordings = manifest["recordings"]
    difficulty_map = assign_difficulties(recordings)
    split_map = stratified_split(recordings, difficulty_map)

    for split in ["train", "validation", "test"]:
        (OUTPUT_DIR / split / "audio").mkdir(parents=True, exist_ok=True)

    csv_rows = []

    for rec in recordings:
        rec_id = rec["id"]
        src_audio = RECORDINGS_DIR / rec_id / "audio.webm"

        if not src_audio.exists():
            print(f"[SKIP] {rec_id} — audio.webm not found")
            continue

        split = split_map[rec_id]
        wav_name = f"{rec_id}.wav"
        dst_audio = OUTPUT_DIR / split / "audio" / wav_name
        txt_path = OUTPUT_DIR / split / "audio" / f"{rec_id}.txt"

        success = convert_webm_to_wav(src_audio, dst_audio)
        txt_path.write_text(rec["scriptText"], encoding="utf-8")

        difficulty = difficulty_map[rec_id]

        csv_rows.append({
            "id": rec_id,
            "split": split,
            "audio_file": str(dst_audio),
            "transcript": rec["scriptText"],
            "category": rec.get("scriptCategory", "general"),
            "duration_seconds": rec.get("durationSeconds", ""),
            "difficulty_level": difficulty,
            "difficulty_description": DIFFICULTY_LABELS[difficulty],
            "converted_ok": success,
            "created_at": rec.get("createdAt", ""),
        })

        status = "OK" if success else "FAIL"
        print(f"[{status}] [{split}] lvl={difficulty} | {rec.get('durationSeconds', '?')}s | {rec_id}")

    if not csv_rows:
        print("\nNo recordings were processed.")
        print(f"Check that {RECORDINGS_DIR.resolve()} contains UUID-named subfolders with audio.webm inside.")
        return

    csv_path = OUTPUT_DIR / "metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nDone. {len(csv_rows)} recordings processed.")
    for split in ["train", "validation", "test"]:
        rows = [r for r in csv_rows if r["split"] == split]
        lvl_counts = {1: 0, 2: 0, 3: 0}
        for r in rows:
            lvl_counts[r["difficulty_level"]] += 1
        print(f"  {split:12s}: {len(rows):3d} total  |  lvl1={lvl_counts[1]}  lvl2={lvl_counts[2]}  lvl3={lvl_counts[3]}")

    total_duration = sum(
        r["duration_seconds"] for r in csv_rows
        if isinstance(r["duration_seconds"], (int, float))
    )
    print(f"  Total audio: {total_duration / 60:.1f} minutes")
    print(f"CSV saved to: {csv_path}")

if __name__ == "__main__":
    build_dataset()