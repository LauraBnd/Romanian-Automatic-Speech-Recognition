import csv
import random
import torch
from pathlib import Path
from datasets import Dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)
import evaluate
from tqdm import tqdm
from transformers.utils import logging
import matplotlib.pyplot as plt

logging.set_verbosity_error()

MODEL_PATH = "stage4_model/final"
FINAL_CSV = Path("stage3_dataset/final_manifest.csv")

LANGUAGE = "romanian"
TASK = "transcribe"
SAMPLE_RATE = 16000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)

if DEVICE == "cuda":
    print(torch.cuda.get_device_name(0))
    torch.cuda.empty_cache()

print("Loading model...")

processor = WhisperProcessor.from_pretrained(MODEL_PATH)

model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
model = model.to(DEVICE)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

model.generation_config.language = LANGUAGE
model.generation_config.task = TASK
model.generation_config.forced_decoder_ids = None
model.generation_config.suppress_tokens = []

model.eval()

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

rows = []

print("Loading testing set...")

with open(FINAL_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        if row["split"] == "validation":
            rows.append({
                "audio": row["final_file"],
                "sentence": row["transcript_normalized"],
            })

dataset = Dataset.from_dict({
    "audio": [r["audio"] for r in rows],
    "sentence": [r["sentence"] for r in rows],
}).cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

print(f"Testing samples: {len(dataset)}")

predictions = []
references = []

durations = []
sample_wers = []

results = []

BATCH_SIZE = 8

for start_idx in tqdm(range(0, len(dataset), BATCH_SIZE)):

    batch = dataset[start_idx:start_idx + BATCH_SIZE]

    audio_arrays = [x["array"] for x in batch["audio"]]

    inputs = processor(
        audio_arrays,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    )

    input_features = inputs.input_features.to(DEVICE)
    attention_mask = inputs.attention_mask.to(DEVICE)

    with torch.no_grad():

        predicted_ids = model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            max_new_tokens=128,
            do_sample=False,
            use_cache=True,
        )

    transcriptions = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    for i in range(len(transcriptions)):

        ref = batch["sentence"][i]
        hyp = transcriptions[i]

        wer = wer_metric.compute(
            predictions=[hyp],
            references=[ref]
        ) * 100

        audio = batch["audio"][i]
        duration = len(audio["array"]) / SAMPLE_RATE

        sample_wers.append(wer)
        durations.append(duration)

        results.append({
            "reference": ref,
            "prediction": hyp,
            "wer": wer,
            "duration": duration
        })

    predictions.extend(transcriptions)
    references.extend(batch["sentence"])

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

final_wer = 100 * wer_metric.compute(
    predictions=predictions,
    references=references
)

final_cer = 100 * cer_metric.compute(
    predictions=predictions,
    references=references
)

print("\nFINAL WER:", round(final_wer, 2), "%")
print("FINAL CER:", round(final_cer, 2), "%")


best_predictions = sorted(results, key=lambda x: x["wer"])[:5]

worst_predictions = sorted(
    results,
    key=lambda x: x["wer"],
    reverse=True
)[:5]

used_ids = set()

for x in best_predictions:
    used_ids.add(id(x))

for x in worst_predictions:
    used_ids.add(id(x))

remaining = [x for x in results if id(x) not in used_ids]

random.shuffle(remaining)

other_predictions = remaining[:10]

output_file = "prediction_examples.txt"

with open(output_file, "w", encoding="utf-8") as f:

    f.write("FINAL RESULTS\n")
    f.write(f"WER: {round(final_wer, 2)}%\n")
    f.write(f"CER: {round(final_cer, 2)}%\n\n")

    f.write("TOP 5 BEST PREDICTIONS\n")

    for idx, item in enumerate(best_predictions, 1):

        f.write(f"Example #{idx}\n")
        f.write(f"WER       : {item['wer']:.2f}%\n")
        f.write(f"REFERENCE : {item['reference']}\n")
        f.write(f"PREDICTED : {item['prediction']}\n")
        f.write("\n" + "-" * 60 + "\n\n")

    f.write("TOP 5 WORST PREDICTIONS\n")
    for idx, item in enumerate(worst_predictions, 1):

        f.write(f"Example #{idx}\n")
        f.write(f"WER       : {item['wer']:.2f}%\n")
        f.write(f"REFERENCE : {item['reference']}\n")
        f.write(f"PREDICTED : {item['prediction']}\n")
        f.write("\n" + "-" * 60 + "\n\n")

    f.write("10 OTHER RANDOM PREDICTIONS\n")

    for idx, item in enumerate(other_predictions, 1):

        f.write(f"Example #{idx}\n")
        f.write(f"WER       : {item['wer']:.2f}%\n")
        f.write(f"REFERENCE : {item['reference']}\n")
        f.write(f"PREDICTED : {item['prediction']}\n")
        f.write("\n" + "-" * 60 + "\n\n")

print(f"\nSaved examples to: {output_file}")

plt.figure()
plt.scatter(durations, sample_wers)
plt.xlabel("Audio duration (seconds)")
plt.ylabel("WER (%)")
plt.title("WER vs Audio Duration")
plt.show()