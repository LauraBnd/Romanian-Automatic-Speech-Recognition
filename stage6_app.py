import os, sys, subprocess, struct, pickle, threading, traceback

def run_worker():
    import gc, torch, numpy as np
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from transformers.utils import logging
    from math import gcd

    os.environ["OMP_NUM_THREADS"]        = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    MODEL_PATH         = "xxx_model_save/2/final"
    DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
    LANGUAGE           = "romanian"
    TASK               = "transcribe"
    SAMPLE_RATE        = 16000
    WHISPER_MEL_LENGTH = 3000

    logging.set_verbosity_error()

    def log(msg):
        print(msg, file=sys.stderr, flush=True)

    log(f"[worker] Device: {DEVICE}")

    processor = WhisperProcessor.from_pretrained(MODEL_PATH)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )
    model.to(DEVICE)
    model.eval()

    model.config.forced_decoder_ids            = None
    model.config.suppress_tokens               = []
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens    = []
    model.generation_config.max_length         = None

    try:
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)
    except Exception:
        forced_decoder_ids = None

    stdin  = sys.stdin.buffer
    stdout = sys.stdout.buffer

    def recv():
        raw = stdin.read(4)
        if len(raw) < 4:
            return None
        (n,) = struct.unpack("<I", raw)
        return pickle.loads(stdin.read(n))

    def send(obj):
        data = pickle.dumps(obj, protocol=4)
        stdout.write(struct.pack("<I", len(data)) + data)
        stdout.flush()

    log("[worker] Model loaded. Sending ready signal...")
    send(("ready", ""))

    while True:
        msg = recv()
        if msg is None:
            break

        sr, samples = msg
        log(f"[worker] Got audio: sr={sr}, shape={samples.shape}")
        try:
            if samples.dtype.kind == "i":
                samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max
            else:
                samples = samples.astype(np.float32)

            if samples.ndim == 2:
                samples = samples.mean(axis=1)

            if sr != SAMPLE_RATE:
                try:
                    from scipy.signal import resample_poly
                    g       = gcd(SAMPLE_RATE, sr)
                    samples = resample_poly(samples, SAMPLE_RATE // g, sr // g).astype(np.float32)
                except ImportError:
                    n_out   = int(len(samples) * SAMPLE_RATE / sr)
                    xs      = np.linspace(0, len(samples) - 1, len(samples))
                    xs_out  = np.linspace(0, len(samples) - 1, n_out)
                    samples = np.interp(xs_out, xs, samples).astype(np.float32)

            if len(samples) == 0:
                send(("ok", "Empty audio — please try again."))
                continue

            feat = processor(samples, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
            T = feat.shape[-1]
            if T < WHISPER_MEL_LENGTH:
                pad  = torch.zeros(1, feat.shape[1], WHISPER_MEL_LENGTH - T)
                feat = torch.cat([feat, pad], dim=-1)
            else:
                feat = feat[..., :WHISPER_MEL_LENGTH]

            dtype          = next(model.parameters()).dtype
            input_features = feat.to(device=DEVICE, dtype=dtype)
            attention_mask = torch.ones(input_features.shape[:-1], dtype=torch.long, device=DEVICE)

            gen_kwargs = dict(
                input_features=input_features,
                attention_mask=attention_mask,
                max_new_tokens=128,
                use_cache=True,
                do_sample=False,
            )
            if forced_decoder_ids is not None:
                gen_kwargs["forced_decoder_ids"] = forced_decoder_ids

            with torch.no_grad():
                predicted_ids = model.generate(**gen_kwargs)

            text = processor.batch_decode(
                predicted_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

            log(f"[worker] Transcription: {text!r}")
            send(("ok", text or "(no speech detected)"))

        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            send(("err", str(exc)))
        finally:
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            gc.collect()


if os.environ.get("ASR_WORKER") == "1":
    run_worker()
    sys.exit(0)


import gradio as gr

_worker_proc = None
_worker_lock = threading.Lock()


def _send(proc, obj):
    data = pickle.dumps(obj, protocol=4)
    proc.stdin.write(struct.pack("<I", len(data)) + data)
    proc.stdin.flush()


def _recv(proc):
    raw = proc.stdout.read(4)
    if len(raw) < 4:
        return None
    (n,) = struct.unpack("<I", raw)
    return pickle.loads(proc.stdout.read(n))


def _start_worker():
    env = os.environ.copy()
    env["ASR_WORKER"] = "1"
    proc = subprocess.Popen(
        [sys.executable, __file__],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        env=env,
    )
    print("[main] Waiting for worker to load model...", flush=True)
    reply = _recv(proc)
    if reply is None or reply[0] != "ready":
        proc.kill()
        raise RuntimeError(f"Worker failed to start, got: {reply}")
    print("[main] Worker ready.", flush=True)
    return proc


def _get_worker():
    global _worker_proc
    if _worker_proc is None or _worker_proc.poll() is not None:
        print("[main] (Re)starting inference worker...", flush=True)
        _worker_proc = _start_worker()
    return _worker_proc


def transcribe(audio):
    global _worker_proc
    if audio is None:
        return "No audio provided."

    with _worker_lock:
        worker = _get_worker()
        try:
            _send(worker, audio)
            reply = _recv(worker)
        except Exception as exc:
            traceback.print_exc()
            _worker_proc = None
            return f"Worker error: {exc}"

        if reply is None:
            _worker_proc = None
            return "Worker died — it will restart on the next request."

        status, text = reply
        if status == "err":
            return f"Transcription error: {text}"
        return text


demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources=["microphone", "upload"], type="numpy", label="Record or Upload Audio"),
    outputs=gr.Textbox(label="Romanian Transcription", lines=6),
    title="Romanian Whisper ASR",
)

with _worker_lock:
    _worker_proc = _start_worker()

demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, share=False, max_threads=1)