import json
import csv
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from functools import partial

from datasets import Dataset, DatasetDict, Audio, load_from_disk
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate

FINAL_CSV = Path("stage3_dataset/final_manifest.csv")

MODEL_NAME = "openai/whisper-small"

OUTPUT_DIR = Path("stage4_model")
CACHE_DIR = Path("stage4_cache")
LOG_DIR = Path("stage4_logs")

LANGUAGE = "romanian"
TASK = "transcribe"
SAMPLE_RATE = 16000

torch.backends.cudnn.benchmark = True


def load_splits():
    splits = {"train": [], "validation": [], "test": []}

    with open(FINAL_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            if split in splits:
                splits[split].append({
                    "audio": row["final_file"],
                    "sentence": row["transcript_normalized"],
                })

    return splits


def make_dataset(rows):
    return Dataset.from_dict({
        "audio": [r["audio"] for r in rows],
        "sentence": [r["sentence"] for r in rows],
    }).cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))


@dataclass
class Collator:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        labels = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(labels, return_tensors="pt")

        labels_tensor = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels_tensor
        return batch


def preprocess(batch, feature_extractor, tokenizer):
    audio = batch["audio"]

    batch["input_features"] = feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    batch["labels"] = tokenizer(batch["sentence"]).input_ids

    return batch


if __name__ == "__main__":

    OUTPUT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)

    if device == "cuda":
        print(torch.cuda.get_device_name(0))

    print("Loading dataset...")

    splits = load_splits()

    dataset = DatasetDict({
        "train": make_dataset(splits["train"]),
        "validation": make_dataset(splits["validation"]),
        "test": make_dataset(splits["test"]),
    })

    print("Loading model...")

    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)

    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model = model.to(device)

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None

    print("Preparing dataset...")

    if CACHE_DIR.exists():
        dataset = load_from_disk(str(CACHE_DIR))
        print("Loaded cached dataset")
    else:
        fn = partial(
            preprocess,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer
        )

        dataset = dataset.map(
            fn,
            remove_columns=["audio", "sentence"]
        )

        dataset.save_to_disk(str(CACHE_DIR))
        print("Dataset cached")

    wer = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        return {"wer": 100 * wer.compute(predictions=pred_str, references=label_str)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),

        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,

        learning_rate=1e-5,
        warmup_steps=100,
        num_train_epochs=6,

        fp16=True,

        eval_steps=1000,
        save_steps=1000,
        logging_steps=50,

        predict_with_generate=False,

        load_best_model_at_end=False,

        report_to="none",
        dataloader_num_workers=0,
    )

    collator = Collator(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    print("\nSTART TRAINING")

    trainer.train()

    print("\nSaving model...")

    save_path = OUTPUT_DIR / "final"
    trainer.save_model(str(save_path))
    processor.save_pretrained(str(save_path))

    print("Saved to:", save_path)

    print("\nEvaluating...")

    results = trainer.evaluate()

    print(results)

    with open(LOG_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nDONE")