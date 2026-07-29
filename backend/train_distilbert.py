"""Fine-tune DistilBERT for cognitive-distortion classification.

This is the reproducible implementation of experiment 7 in ML_EXPERIMENTS.md.
This is the fourth iteration of the ML pipeline, currently deployed in app.py, replacing earlier MiniLM + LogisticRegression pipeline. The trained tokenizer, model, and label mappings are saved together in
Hugging Face's ``save_pretrained`` format so they can be loaded for inference later.

Run from any directory:

    python3 backend/train_distilbert.py

The first run downloads ``distilbert-base-uncased`` from Hugging Face. Training is
automatically accelerated by CUDA or Apple Silicon MPS when available.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "distilbert-base-uncased"
DEFAULT_OUTPUT_DIR = (
    BASE_DIR / "models" / "distilbert-cognitive-distortions-improved"
)
RANDOM_SEED = 42


def load_dataset() -> pd.DataFrame:
    """Build the same 3,526-row dataset used by the existing model experiments."""
    kaggle_df = pd.read_csv(BASE_DIR / "cognitive_distortion_dataset.csv")
    kaggle_df = kaggle_df.dropna(
        subset=["Patient Question", "Dominant Distortion"]
    )
    augmented_df = pd.read_csv(BASE_DIR / "augmented_data.csv")

    no_distortion = kaggle_df[
        kaggle_df["Dominant Distortion"] == "No Distortion"
    ]
    has_distortion = kaggle_df[
        kaggle_df["Dominant Distortion"] != "No Distortion"
    ]

    has_distortion_clean = has_distortion[
        ["Distorted part", "Dominant Distortion"]
    ].copy()
    has_distortion_clean.columns = ["text", "label"]
    has_distortion_clean = has_distortion_clean.dropna(subset=["text", "label"])
    # Each row here already comes from a distinct source paragraph, so there's no
    # shared-paragraph leakage risk — each gets its own group.
    has_distortion_clean["group"] = [
        f"has_distortion_{i}" for i in has_distortion_clean.index
    ]

    no_distortion_sentences = []
    for idx, paragraph in no_distortion["Patient Question"].items():
        sentences = re.split(r"(?<=[.!?])\s+", paragraph.strip())
        # Tag every sentence with the paragraph it came from so the split can keep
        # all of a paragraph's sentences on the same side — otherwise sentences from
        # the same writer/story (overlapping vocabulary and tone) could land on both
        # sides purely by chance.
        no_distortion_sentences.extend(
            {
                "text": sentence.strip(),
                "label": "No Distortion",
                "group": f"no_distortion_paragraph_{idx}",
            }
            for sentence in sentences
            if sentence.strip()
        )

    no_distortion_clean = pd.DataFrame(no_distortion_sentences)
    no_distortion_downsampled = no_distortion_clean.sample(
        n=400, random_state=RANDOM_SEED
    )

    augmented_df = augmented_df.copy()
    augmented_df["group"] = [f"augmented_{i}" for i in augmented_df.index]

    dataset = pd.concat(
        [has_distortion_clean, no_distortion_downsampled, augmented_df],
        ignore_index=True,
    )
    dataset = dataset.dropna(subset=["text", "label"])
    dataset["text"] = dataset["text"].astype(str).str.strip()
    dataset["label"] = dataset["label"].astype(str).str.strip()
    dataset = dataset[dataset["text"] != ""]

    # Repeated examples can otherwise land on both sides of the split and inflate
    # evaluation metrics without teaching the model anything new.
    return dataset.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)


class EncodedTextDataset:
    """Minimal map-style dataset accepted by Hugging Face Trainer."""

    def __init__(self, encodings: dict[str, list[list[int]]], labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = {key: values[index] for key, values in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT on the cognitive-distortion dataset."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=float, default=6.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Checkpoint directory, or 'true' to use the latest checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    # Imports live here so dataset preparation and static checks do not initialize
    # PyTorch or the Hugging Face training stack.
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    args = parse_args()
    set_seed(RANDOM_SEED)

    dataframe = load_dataset()
    labels = sorted(dataframe["label"].unique().tolist())
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}

    # Keep the final test set untouched while Trainer uses validation metrics to
    # choose a checkpoint. The old implementation selected and reported on the
    # same holdout, which made its final score optimistically biased.
    #
    # GroupShuffleSplit (instead of a plain stratified train_test_split) keeps every
    # sentence from the same source "No Distortion" paragraph on one side of each
    # split, so validation/test are never evaluated on sentences whose
    # vocabulary/tone the model already saw in train via a sibling sentence from the
    # same paragraph. GroupShuffleSplit doesn't support stratify, so class balance
    # across the splits is no longer guaranteed exactly, but it wasn't safe to trade
    # away leakage-freedom for it.
    texts = dataframe["text"].to_numpy()
    label_ids = dataframe["label"].map(label2id).to_numpy()
    groups = dataframe["group"].to_numpy()

    train_val_split = GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=RANDOM_SEED
    )
    train_idx, holdout_idx = next(
        train_val_split.split(texts, label_ids, groups=groups)
    )

    val_test_split = GroupShuffleSplit(
        n_splits=1, test_size=0.5, random_state=RANDOM_SEED
    )
    val_idx_in_holdout, test_idx_in_holdout = next(
        val_test_split.split(
            holdout_idx, label_ids[holdout_idx], groups=groups[holdout_idx]
        )
    )
    validation_idx = holdout_idx[val_idx_in_holdout]
    test_idx = holdout_idx[test_idx_in_holdout]

    train_texts = texts[train_idx].tolist()
    train_labels = label_ids[train_idx].tolist()
    validation_texts = texts[validation_idx].tolist()
    validation_labels = label_ids[validation_idx].tolist()
    test_texts = texts[test_idx].tolist()
    test_labels = label_ids[test_idx].tolist()

    print(f"Dataset size: {len(dataframe):,}")
    print(f"Training rows: {len(train_texts):,}")
    print(f"Validation rows: {len(validation_texts):,}")
    print(f"Test rows: {len(test_texts):,}")
    print(f"Labels ({len(labels)}): {labels}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_encodings = tokenizer(
        train_texts, truncation=True, max_length=args.max_length
    )
    validation_encodings = tokenizer(
        validation_texts, truncation=True, max_length=args.max_length
    )
    test_encodings = tokenizer(
        test_texts, truncation=True, max_length=args.max_length
    )
    train_dataset = EncodedTextDataset(train_encodings, train_labels)
    validation_dataset = EncodedTextDataset(
        validation_encodings, validation_labels
    )
    test_dataset = EncodedTextDataset(test_encodings, test_labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    def compute_metrics(eval_prediction) -> dict[str, float]:
        predictions = np.argmax(eval_prediction.predictions, axis=-1)
        return {
            "accuracy": accuracy_score(eval_prediction.label_ids, predictions),
            "f1_macro": f1_score(
                eval_prediction.label_ids,
                predictions,
                average="macro",
                zero_division=0,
            ),
        }

    output_dir = args.output_dir.resolve()
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_steps=round(
            math.ceil(len(train_dataset) / args.batch_size)
            * args.epochs
            * args.warmup_ratio
        ),
        label_smoothing_factor=args.label_smoothing,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        seed=RANDOM_SEED,
        data_seed=RANDOM_SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        ],
    )

    resume = args.resume_from_checkpoint
    if isinstance(resume, str) and resume.lower() == "true":
        resume = True
    trainer.train(resume_from_checkpoint=resume)

    prediction_output = trainer.predict(test_dataset)
    predicted_ids = np.argmax(prediction_output.predictions, axis=-1)
    accuracy = accuracy_score(test_labels, predicted_ids)
    macro_f1 = f1_score(
        test_labels, predicted_ids, average="macro", zero_division=0
    )
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    print(
        classification_report(
            test_labels,
            predicted_ids,
            labels=list(range(len(labels))),
            target_names=labels,
            zero_division=0,
        )
    )

    # save_pretrained stores the classification head and label mappings in config.json.
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "base_model": args.model,
                "dataset_size": len(dataframe),
                "train_size": len(train_texts),
                "validation_size": len(validation_texts),
                "test_size": len(test_texts),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "best_validation_macro_f1": trainer.state.best_metric,
                "best_checkpoint": (
                    Path(trainer.state.best_model_checkpoint).name
                    if trainer.state.best_model_checkpoint
                    else None
                ),
                "labels": labels,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "max_length": args.max_length,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "label_smoothing": args.label_smoothing,
                "early_stopping_patience": args.early_stopping_patience,
                "seed": RANDOM_SEED,
            },
            file,
            indent=2,
        )
    print(f"Model and tokenizer saved to {output_dir}")


if __name__ == "__main__":
    main()
