import csv
import json
import re
import shutil
import unicodedata
from pathlib import Path
from collections import defaultdict

STAGE1_CSV = Path("stage1_dataset/metadata.csv")
STAGE2_CSV = Path("stage2_opensource/manifest.csv")
OUTPUT_DIR = Path("stage3_dataset")
REPORT_PATH = Path("stage3_dataset/verification_report.json")
FINAL_CSV = Path("stage3_dataset/final_manifest.csv")

MIN_DURATION = 1.0
MAX_DURATION = 30.0

RO_NUM_MAP = {
    "0": "zero", "1": "unu", "2": "doi", "3": "trei", "4": "patru",
    "5": "cinci", "6": "șase", "7": "șapte", "8": "opt", "9": "nouă",
    "10": "zece", "11": "unsprezece", "12": "doisprezece", "13": "treisprezece",
    "14": "paisprezece", "15": "cincisprezece", "16": "șaisprezece",
    "17": "șaptesprezece", "18": "optsprezece", "19": "nouăsprezece",
    "20": "douăzeci", "30": "treizeci", "40": "patruzeci", "50": "cincizeci",
    "100": "o sută", "1000": "o mie",
}

def normalize_text(text: str) -> str:
    text = text.strip()
    text = text.lower()
    text = unicodedata.normalize("NFC", text)

    def replace_number(m):
        n = m.group(0)
        return RO_NUM_MAP.get(n, n)

    text = re.sub(r"\b\d+\b", replace_number, text)
    text = re.sub(r"[^\w\s\-]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_stage1():
    rows = []
    with open(STAGE1_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "id": row["id"],
                "split": row["split"],
                "file": row["audio_file"],
                "transcript": row["transcript"],
                "duration_seconds": float(row["duration_seconds"]) if row["duration_seconds"] else 0.0,
                "source": f"stage1_{row.get('category', 'personal')}",
                "difficulty": row.get("difficulty_level", ""),
                "converted_ok": row.get("converted_ok", "True"),
            })
    return rows

def load_stage2():
    rows = []
    with open(STAGE2_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            rows.append({
                "id": f"s2_{i:06d}",
                "split": None,
                "file": row["file"],
                "transcript": row["transcript"],
                "duration_seconds": float(row["duration_seconds"]),
                "source": row["source"],
                "difficulty": "",
                "converted_ok": "True",
            })
    return rows

def stratified_split_stage2(rows):
    import random
    rng = random.Random(42)
    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)

    for group in by_source.values():
        rng.shuffle(group)
        n = len(group)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)
        for i, r in enumerate(group):
            if i < train_end:
                r["split"] = "train"
            elif i < val_end:
                r["split"] = "validation"
            else:
                r["split"] = "test"

def filter_and_normalize(rows):
    kept, filtered = [], []
    for r in rows:
        reasons = []

        if r["converted_ok"] == "False":
            reasons.append("conversion_failed")

        filepath = Path(r["file"])
        if not filepath.exists():
            reasons.append("file_missing")

        duration = r["duration_seconds"]
        if duration < MIN_DURATION:
            reasons.append(f"too_short_{duration:.2f}s")
        if duration > MAX_DURATION:
            reasons.append(f"too_long_{duration:.2f}s")

        transcript = r["transcript"].strip()
        if not transcript:
            reasons.append("empty_transcript")

        if reasons:
            filtered.append({**r, "reasons": reasons})
        else:
            r["transcript_raw"] = transcript
            r["transcript_normalized"] = normalize_text(transcript)
            kept.append(r)

    return kept, filtered

def copy_to_output(rows):
    for split in ["train", "validation", "test"]:
        (OUTPUT_DIR / split / "audio").mkdir(parents=True, exist_ok=True)

    for r in rows:
        src = Path(r["file"])
        dst = OUTPUT_DIR / r["split"] / "audio" / src.name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

        txt_dst = dst.with_suffix(".txt")
        txt_dst.write_text(r["transcript_normalized"], encoding="utf-8")

        r["final_file"] = str(dst)

print("Loading Stage 1 data...")
s1 = load_stage1()
print(f"  {len(s1)} personal recordings")

print("Loading Stage 2 data...")
s2 = load_stage2()
print(f"  {len(s2)} open-source recordings")
stratified_split_stage2(s2)

all_rows = s1 + s2
print(f"\nTotal before filtering: {len(all_rows)}")

kept, filtered = filter_and_normalize(all_rows)
print(f"Kept   : {len(kept)}")
print(f"Removed: {len(filtered)}")

print("\nCopying files to stage3_dataset/...")
copy_to_output(kept)

with open(FINAL_CSV, "w", newline="", encoding="utf-8") as f:
    fieldnames = ["id", "split", "final_file", "transcript_raw",
                  "transcript_normalized", "duration_seconds", "source", "difficulty"]
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(kept)

by_split = defaultdict(lambda: {"count": 0, "hours": 0.0})
by_source = defaultdict(lambda: {"count": 0, "hours": 0.0})
for r in kept:
    by_split[r["split"]]["count"] += 1
    by_split[r["split"]]["hours"] += r["duration_seconds"] / 3600
    by_source[r["source"]]["count"] += 1
    by_source[r["source"]]["hours"] += r["duration_seconds"] / 3600

report = {
    "total_kept": len(kept),
    "total_filtered": len(filtered),
    "filter_criteria": {
        "min_duration_seconds": MIN_DURATION,
        "max_duration_seconds": MAX_DURATION,
        "require_transcript": True,
        "require_file_exists": True,
        "require_conversion_ok": True,
    },
    "normalization": {
        "lowercase": True,
        "unicode_NFC": True,
        "numbers_to_words": True,
        "punctuation_removed": True,
    },
    "by_split": {k: {"count": v["count"], "hours": round(v["hours"], 3)}
                 for k, v in by_split.items()},
    "by_source": {k: {"count": v["count"], "hours": round(v["hours"], 3)}
                  for k, v in by_source.items()},
    "filtered_entries": [{"id": r["id"], "file": r["file"], "reasons": r["reasons"]}
                         for r in filtered],
}

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"{'Split':<14} {'Clips':>6} {'Hours':>8}")
for split in ["train", "validation", "test"]:
    s = by_split[split]
    print(f"{split:<14} {s['count']:>6} {s['hours']:>8.3f}h")
total_h = sum(v["hours"] for v in by_split.values())
total_c = sum(v["count"] for v in by_split.values())
print(f"{'TOTAL':<14} {total_c:>6} {total_h:>8.3f}h")
print(f"\nFiltered out: {len(filtered)} clips")
if filtered:
    reason_counts = defaultdict(int)
    for r in filtered:
        for reason in r["reasons"]:
            reason_counts[reason] += 1
    for reason, count in reason_counts.items():
        print(f"  {reason}: {count}")
print(f"\nReport : {REPORT_PATH}")
print(f"CSV    : {FINAL_CSV}")