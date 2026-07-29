import re

import pandas as pd


def build_training_dataset(include_feedback=False):
    """Build the sentence-level training dataset shared by the embeddings + LR
    scripts (train_embeddings_lr.py, tune_embeddings_lr.py).

    Combines the Kaggle "Distorted part" rows, sentence-split "No Distortion"
    paragraphs (downsampled to 400), and the LLM-augmented rows. Every row is
    tagged with a `group` (its source paragraph) so a group-aware split can
    keep every sentence from the same source paragraph on one side — otherwise
    sentences from the same writer/story (overlapping vocabulary and tone)
    could land on both sides of a split purely by chance.

    Returns (df, feedback_data), where feedback_data is None if
    include_feedback is False, or the raw feedback rows otherwise (the
    caller needs the raw rows separately to mark them used after training).
    """
    kaggle_df = pd.read_csv("cognitive_distortion_dataset.csv")
    kaggle_df = kaggle_df.dropna(subset=["Patient Question", "Dominant Distortion"])

    augmented_df = pd.read_csv("augmented_data.csv")

    no_distortion = kaggle_df[kaggle_df["Dominant Distortion"] == "No Distortion"]
    has_distortion = kaggle_df[kaggle_df["Dominant Distortion"] != "No Distortion"]

    # Each row here already comes from a distinct source paragraph (the "Distorted part"
    # span), so there's no shared-paragraph leakage risk — each gets its own group.
    has_distortion_clean = has_distortion[["Distorted part", "Dominant Distortion"]].copy()
    has_distortion_clean.columns = ["text", "label"]
    has_distortion_clean["group"] = [f"has_distortion_{i}" for i in has_distortion_clean.index]

    no_distortion_sentences = []
    for idx, row in no_distortion.iterrows():
        sentences = re.split(r'(?<=[.!?])\s+', row["Patient Question"].strip())
        for sentence in sentences:
            if sentence.strip():
                no_distortion_sentences.append({
                    "text": sentence.strip(),
                    "label": "No Distortion",
                    "group": f"no_distortion_paragraph_{idx}",
                })

    no_distortion_clean = pd.DataFrame(no_distortion_sentences)
    # Downsample "No Distortion" to ~400 to mitigate class imbalance
    no_distortion_downsampled = no_distortion_clean.sample(n=400, random_state=42)

    kaggle_clean_df = pd.concat([has_distortion_clean, no_distortion_downsampled])

    # combine with augmented data (each augmented example is its own group, same reasoning
    # as has_distortion_clean above)
    augmented_df = augmented_df.copy()
    augmented_df["group"] = [f"augmented_{i}" for i in augmented_df.index]
    df = pd.concat([kaggle_clean_df, augmented_df])

    if not include_feedback:
        return df, None

    from database import get_training_feedback

    feedback_data = get_training_feedback()
    if len(feedback_data) > 0:
        feedback_df = pd.DataFrame(feedback_data, columns=["text", "label"])
        feedback_df["group"] = [f"feedback_{i}" for i in feedback_df.index]
        df = pd.concat([df, feedback_df])

    return df, feedback_data
