import os, sys, json, csv, io, time, subprocess, tempfile, traceback
from pathlib import Path
import numpy as np
import torch
import evaluate

CSV_PATH       = Path("stage3_dataset/final_manifest.csv")
FINETUNED_PATH = Path("stage4_model/final")
ELEVENLABS_KEY = ""

SAMPLE_RATE  = 16000
BATCH_SIZE   = 1
MAX_SAMPLES  = 50

WHISPER_MODELS = [
    "openai/whisper-tiny",
    "openai/whisper-base",
    "openai/whisper-small",
    "openai/whisper-medium",
    "openai/whisper-large-v3",
]

MODEL_TIMEOUTS = {
    "openai/whisper-tiny":     300,
    "openai/whisper-base":     300,
    "openai/whisper-small":    600,
    "openai/whisper-medium":   900,
    "openai/whisper-large-v3": 1800,
    "fine-tuned":              600,
    "elevenlabs":              600,
}
DEFAULT_TIMEOUT = 1800

def _worker_main():
    payload_path = Path(sys.argv[2])
    payload      = json.loads(payload_path.read_text())

    model_type  = payload["type"]
    model_name  = payload.get("model", "")
    data_file   = Path(payload["data_file"])
    result_file = Path(payload["result_file"])

    os.environ["OMP_NUM_THREADS"]        = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[WORKER] Device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else " (no GPU found, using CPU)"))

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def score(preds, refs):
        preds = [p.strip() for p in preds]
        refs  = [r.strip() for r in refs]
        wer = 100 * wer_metric.compute(predictions=preds, references=refs)
        cer = 100 * cer_metric.compute(predictions=preds, references=refs)
        return round(wer, 2), round(cer, 2)

    data = json.loads(data_file.read_text(encoding="utf-8"))

    if model_type in ("whisper", "finetuned"):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        path = model_name if model_type == "whisper" else payload["finetuned_path"]
        print(f"[WORKER] Loading processor from {path}")

        processor = WhisperProcessor.from_pretrained(path)
        model = WhisperForConditionalGeneration.from_pretrained(
            path, torch_dtype=torch.float32
        ).to(device)

        model.config.forced_decoder_ids = None
        model.config.suppress_tokens    = []
        model.generation_config.forced_decoder_ids = None
        model.eval()

        WHISPER_MEL_LENGTH = 3000

        try:
            forced_decoder_ids = processor.get_decoder_prompt_ids(
                language="romanian", task="transcribe"
            )
        except Exception:
            forced_decoder_ids = None

        preds, refs = [], []
        for i in range(0, len(data), BATCH_SIZE):
            chunk = data[i : i + BATCH_SIZE]
            audio_arrays = [np.array(s["audio"], dtype=np.float32) for s in chunk]

            mel_list = []
            for audio in audio_arrays:
                feat = processor(
                    audio,
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt",
                ).input_features
                T = feat.shape[-1]
                if T < WHISPER_MEL_LENGTH:
                    pad  = torch.zeros(1, feat.shape[1], WHISPER_MEL_LENGTH - T)
                    feat = torch.cat([feat, pad], dim=-1)
                else:
                    feat = feat[..., :WHISPER_MEL_LENGTH]
                mel_list.append(feat)

            input_features = torch.cat(mel_list, dim=0).to(device)

            gen_kwargs = dict(input_features=input_features, max_new_tokens=128)
            if forced_decoder_ids is not None:
                gen_kwargs["forced_decoder_ids"] = forced_decoder_ids

            with torch.no_grad():
                out = model.generate(**gen_kwargs)

            texts = processor.batch_decode(out, skip_special_tokens=True)
            preds.extend(texts)
            refs.extend([s["text"] for s in chunk])
            print(f"  [{i + len(chunk)}/{len(data)}] pred: {texts[0][:70]}")

        wer, cer = score(preds, refs)
        result = {
            "model": model_name or "fine-tuned",
            "type":  model_type,
            "wer":   wer,
            "cer":   cer,
            "preds": preds,
            "refs":  refs,
        }

    elif model_type == "elevenlabs":
        import requests, soundfile as sf

        headers = {"xi-api-key": payload["api_key"]}
        preds, refs = [], []

        for i, sample in enumerate(data):
            audio_np = np.array(sample["audio"], dtype=np.float32)
            buf = io.BytesIO()
            sf.write(buf, audio_np, SAMPLE_RATE, format="WAV")
            buf.seek(0)
            wav_bytes = buf.read()

            try:
                r = requests.post(
                    "https://api.elevenlabs.io/v1/speech-to-text",
                    headers=headers,
                    files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                    data={"model_id": "scribe_v1", "language_code": "ro"},
                    timeout=60,
                )
                text = r.json().get("text", "") if r.status_code == 200 else ""
                if r.status_code != 200:
                    print(f"  [EL] HTTP {r.status_code}: {r.text[:120]}")
            except Exception as exc:
                print(f"  [EL] Exception: {exc}")
                text = ""

            preds.append(text)
            refs.append(sample["text"])
            print(f"  [{i+1}/{len(data)}] pred: {text[:70]}")

        wer, cer = score(preds, refs)
        result = {
            "model": "elevenlabs",
            "type":  "elevenlabs",
            "wer":   wer,
            "cer":   cer,
            "preds": preds,
            "refs":  refs,
        }

    else:
        result = {"error": f"Unknown model_type: {model_type}"}

    result_file.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(f"[WORKER] Done -> {result_file}")

def load_audio_data(csv_path: Path, split="test", max_samples=50) -> list:
    print("\n[DATA] Reading CSV ...")
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == split:
                rows.append((r["final_file"], r["transcript_normalized"]))
    rows = rows[:max_samples]
    print(f"[DATA] Found {len(rows)} samples for split='{split}'")

    data, failed = [], 0
    for audio_path, text in rows:
        try:
            try:
                import librosa
                audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            except ImportError:
                import soundfile as sf
                import numpy as np, math
                audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
                if sr != SAMPLE_RATE:
                    ratio  = SAMPLE_RATE / sr
                    n_out  = int(math.ceil(len(audio) * ratio))
                    xs_in  = np.linspace(0, len(audio) - 1, len(audio))
                    xs_out = np.linspace(0, len(audio) - 1, n_out)
                    audio  = np.interp(xs_out, xs_in, audio).astype(np.float32)
            data.append({"audio": audio.tolist(), "text": text})
        except Exception as exc:
            print(f"  [WARN] Could not load {audio_path}: {exc}")
            failed += 1

    print(f"[DATA] Loaded {len(data)} samples ({failed} failed)")
    return data


def run_in_subprocess(payload: dict, timeout: int) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        payload_file = Path(tmp) / "payload.json"
        result_file  = Path(tmp) / "result.json"

        payload["result_file"] = str(result_file)
        payload_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        cmd = [sys.executable, __file__, "--worker", str(payload_file)]
        label = payload.get("model") or payload.get("type")
        print(f"\n[LAUNCH] {label}  (timeout={timeout//60} min) ...")

        t0 = time.time()
        try:
            ret = subprocess.run(cmd, timeout=timeout)
        except subprocess.TimeoutExpired:
            elapsed = round(time.time() - t0, 1)
            print(f"[TIMEOUT] {label} exceeded {timeout}s — skipping")
            return {
                "model":   label,
                "type":    payload.get("type"),
                "error":   f"Timed out after {timeout}s",
                "elapsed": elapsed,
            }
        elapsed = round(time.time() - t0, 1)

        if ret.returncode != 0:
            return {
                "model":   label,
                "type":    payload.get("type"),
                "error":   f"Worker crashed (exit {ret.returncode})",
                "elapsed": elapsed,
            }

        if not result_file.exists():
            return {
                "model":   label,
                "type":    payload.get("type"),
                "error":   "Worker exited OK but wrote no result",
                "elapsed": elapsed,
            }

        result = json.loads(result_file.read_text(encoding="utf-8"))
        result["elapsed"] = elapsed
        return result


def print_results_table(results: list):
    print(f"  {'MODEL':<33} {'WER':>8} {'CER':>8} {'TIME':>8}")
    for r in results:
        name = r.get("model", "?")
        if "error" in r:
            print(f"  {name:<33} {'ERROR':>8}  <- {r['error']}")
        else:
            t = f"{r.get('elapsed', 0):.0f}s"
            print(f"  {name:<33} {r['wer']:>7}% {r['cer']:>7}% {t:>8}")


def save_results(results: list, out_path: Path = Path("benchmark_results.json")):
    safe = [{k: v for k, v in r.items() if k not in ("preds", "refs")} for r in results]
    out_path.write_text(json.dumps(safe, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[SAVED] Summary -> {out_path}")

    detail_path = out_path.with_name("benchmark_predictions.json")
    detail_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[SAVED] Predictions -> {detail_path}")


def main():
    print("\nASR BENCHMARK  (subprocess-isolated, Windows-safe)\n")

    data = load_audio_data(CSV_PATH, split="test", max_samples=MAX_SAMPLES)
    if not data:
        print("[ERROR] No data loaded. Check CSV_PATH and split column.")
        sys.exit(1)

    tmp_data_file = Path(tempfile.mktemp(suffix="_asr_data.json"))
    tmp_data_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"[DATA] Audio cache -> {tmp_data_file}  ({tmp_data_file.stat().st_size // 1024} KB)")

    results = []

    if FINETUNED_PATH.exists():
        results.append(run_in_subprocess(
            {
                "type":           "finetuned",
                "model":          "fine-tuned",
                "finetuned_path": str(FINETUNED_PATH),
                "data_file":      str(tmp_data_file),
            },
            timeout=MODEL_TIMEOUTS.get("fine-tuned", DEFAULT_TIMEOUT),
        ))
    else:
        print(f"\n[SKIP] Fine-tuned model not found at {FINETUNED_PATH}")

    for model_name in WHISPER_MODELS:
        results.append(run_in_subprocess(
            {
                "type":      "whisper",
                "model":     model_name,
                "data_file": str(tmp_data_file),
            },
            timeout=MODEL_TIMEOUTS.get(model_name, DEFAULT_TIMEOUT),
        ))

    if ELEVENLABS_KEY:
        results.append(run_in_subprocess(
            {
                "type":      "elevenlabs",
                "model":     "elevenlabs",
                "api_key":   ELEVENLABS_KEY,
                "data_file": str(tmp_data_file),
            },
            timeout=MODEL_TIMEOUTS.get("elevenlabs", DEFAULT_TIMEOUT),
        ))

    tmp_data_file.unlink(missing_ok=True)

    print_results_table(results)
    save_results(results)

if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        try:
            _worker_main()
        except Exception:
            traceback.print_exc()
            sys.exit(1)
    else:
        main()