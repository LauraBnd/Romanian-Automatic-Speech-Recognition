import json
import csv
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Any
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

RESUME_FROM = Path("stage4_model/final")

OUTPUT_DIR = Path("stage4_continued")
CACHE_DIR  = Path("stage4_cache")
LOG_DIR    = Path("stage4_continued_logs")

LANGUAGE    = "romanian"
TASK        = "transcribe"
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
                    "audio":    row["final_file"],
                    "sentence": row["transcript_normalized"],
                })
    return splits


def make_dataset(rows):
    return Dataset.from_dict({
        "audio":    [r["audio"]    for r in rows],
        "sentence": [r["sentence"] for r in rows],
    }).cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))


def preprocess(batch, feature_extractor, tokenizer):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


@dataclass
class Collator:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        labels       = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(labels, return_tensors="pt")

        labels_tensor = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if not (labels_tensor[:, 0] == self.decoder_start_token_id).all():
            decoder_input_ids = torch.full(
                (labels_tensor.shape[0], 1),
                self.decoder_start_token_id,
                dtype=torch.long,
            )
            labels_tensor = torch.cat([decoder_input_ids, labels_tensor], dim=1)

        batch["labels"] = labels_tensor
        return batch

if __name__ == "__main__":

    OUTPUT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    if device == "cuda":
        print(torch.cuda.get_device_name(0))

    print(f"Loading model from: {RESUME_FROM}")

    feature_extractor = WhisperFeatureExtractor.from_pretrained(str(RESUME_FROM))
    tokenizer  = WhisperTokenizer.from_pretrained(str(RESUME_FROM), language=LANGUAGE, task=TASK)
    processor  = WhisperProcessor.from_pretrained(str(RESUME_FROM), language=LANGUAGE, task=TASK)
    model      = WhisperForConditionalGeneration.from_pretrained(str(RESUME_FROM))
    model      = model.to(device)

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    model.generation_config.language           = LANGUAGE
    model.generation_config.task               = TASK
    model.generation_config.forced_decoder_ids = None

    print("Loading dataset...")

    if CACHE_DIR.exists():
        dataset = load_from_disk(str(CACHE_DIR))
        print("Loaded cached dataset from", CACHE_DIR)
    else:
        splits  = load_splits()
        dataset = DatasetDict({
            "train":      make_dataset(splits["train"]),
            "validation": make_dataset(splits["validation"]),
            "test":       make_dataset(splits["test"]),
        })
        fn = partial(preprocess, feature_extractor=feature_extractor, tokenizer=tokenizer)
        dataset = dataset.map(fn, remove_columns=["audio", "sentence"])
        dataset.save_to_disk(str(CACHE_DIR))
        print("Preprocessed and cached dataset to", CACHE_DIR)

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": 100 * wer_metric.compute(predictions=pred_str, references=label_str)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),

        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,

        learning_rate=1e-7,
        warmup_steps=100,
        num_train_epochs=5,
        lr_scheduler_type="cosine",

        fp16=(device == "cuda"),

        label_smoothing_factor=0.1,

        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=25,

        predict_with_generate=True,
        generation_max_length=225,

        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,

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

    is_hf_checkpoint = (RESUME_FROM / "trainer_state.json").exists()

    print("\nSTART CONTINUED TRAINING")
    if is_hf_checkpoint:
        print(f"Resuming optimizer/scheduler state from: {RESUME_FROM}")
        trainer.train(resume_from_checkpoint=str(RESUME_FROM))
    else:
        print("Starting fresh optimizer (weights-only checkpoint).")
        trainer.train()


    print("\nSaving model...")
    save_path = OUTPUT_DIR / "final"
    trainer.save_model(str(save_path))
    processor.save_pretrained(str(save_path))
    print("Saved to:", save_path)

    print("\nEvaluating all checkpoints and final model...")

    checkpoint_dirs = sorted(
        OUTPUT_DIR.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1])
    )
    candidates = list(checkpoint_dirs) + [save_path]

    all_results = {}

    for ckpt_path in candidates:
        label = ckpt_path.name
        print(f"\n--- Evaluating: {label} ---")
        torch.cuda.empty_cache()

        ckpt_model = WhisperForConditionalGeneration.from_pretrained(str(ckpt_path))
        ckpt_model = ckpt_model.to(device)
        ckpt_model.config.use_cache = True

        ckpt_trainer = Seq2SeqTrainer(
            model=ckpt_model,
            args=training_args,
            eval_dataset=dataset["validation"],
            data_collator=collator,
            compute_metrics=compute_metrics,
            processing_class=processor.feature_extractor,
        )

        ckpt_results = ckpt_trainer.evaluate(metric_key_prefix="eval")
        wer_value = ckpt_results.get("eval_wer", float("inf"))
        all_results[label] = {"path": str(ckpt_path), "wer": wer_value}
        print(f"  WER: {wer_value:.2f}")

        del ckpt_model
        torch.cuda.empty_cache()

    ranked = sorted(all_results.items(), key=lambda x: x[1]["wer"])

    print("CHECKPOINT RANKING (best → worst WER on validation)")
    for rank, (label, info) in enumerate(ranked, 1):
        marker = "  ← BEST" if rank == 1 else ""
        print(f"  {rank:>2}. {label:<30}  WER: {info['wer']:.2f}{marker}")

    best_label, best_info = ranked[0]
    print(f"\nBest model : {best_label}")
    print(f"Path       : {best_info['path']}")
    print(f"WER        : {best_info['wer']:.2f}")

    summary = {
        "ranked": [
            {"rank": i + 1, "checkpoint": label, "path": info["path"], "wer": info["wer"]}
            for i, (label, info) in enumerate(ranked)
        ],
        "best": {"checkpoint": best_label, "path": best_info["path"], "wer": best_info["wer"]},
    }

    with open(LOG_DIR / "checkpoint_ranking.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nFull ranking saved to: {LOG_DIR / 'checkpoint_ranking.json'}")
    print("\nDONE")