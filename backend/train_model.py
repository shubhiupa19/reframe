import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer   
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import re

from database import (
    get_training_feedback,
    mark_used_feedback,
    save_model_version,
    get_latest_version,
)

# Load Kaggle dataset
kaggle_df = pd.read_csv("cognitive_distortion_dataset.csv")

# Drop empty rows and any without a dominant distortion label
kaggle_df = kaggle_df.dropna(subset=["Patient Question", "Dominant Distortion"])

# load augmented data (AI-genereated)
augmented_df = pd.read_csv("augmented_data.csv")

# Split into no distortion vs distortion
no_distortion = kaggle_df[kaggle_df["Dominant Distortion"]== "No Distortion"]
has_distortion = kaggle_df[kaggle_df["Dominant Distortion"] != "No Distortion"]

# for distortion rows, just grab the distorted sentence
has_distortion_clean = has_distortion[["Distorted part", "Dominant Distortion"]].copy()
has_distortion_clean.columns = ["text", "label"]
# Each row here already comes from a distinct source paragraph (the "Distorted part"
# span), so there's no shared-paragraph leakage risk — each gets its own group.
has_distortion_clean["group"] = [f"has_distortion_{i}" for i in has_distortion_clean.index]

# for no distortion, split each paragraph into sentences
no_distortion_sentences = []
for idx, row in no_distortion.iterrows():
    sentences = re.split(r'(?<=[.!?])\s+', row["Patient Question"].strip())
    for sentence in sentences:
        if sentence.strip():
            # Tag every sentence with the paragraph it came from so the train/test
            # split can keep all of a paragraph's sentences on the same side —
            # otherwise sentences from the same writer/story (overlapping
            # vocabulary and tone) could land on both sides purely by chance.
            no_distortion_sentences.append({
                "text": sentence.strip(),
                "label": "No Distortion",
                "group": f"no_distortion_paragraph_{idx}",
            })

no_distortion_clean = pd.DataFrame(no_distortion_sentences)
# Downsample "No Distortion" to ~400 to mitigate class imbalance
no_distortion_downsampled = no_distortion_clean.sample(n=400, random_state=42)

# Store this downsample + rest of data in the primary df
kaggle_clean_df = pd.concat([has_distortion_clean, no_distortion_downsampled])

# combine with augmented data (each augmented example is its own group, same reasoning
# as has_distortion_clean above)
augmented_df = augmented_df.copy()
augmented_df["group"] = [f"augmented_{i}" for i in augmented_df.index]
df = pd.concat([kaggle_clean_df, augmented_df])

# Load user feedback
feedback_data = get_training_feedback()
if len(feedback_data) > 0:
    feedback_df = pd.DataFrame(
        feedback_data, columns=["text", "label"]
    )
    feedback_df["group"] = [f"feedback_{i}" for i in feedback_df.index]
    df = pd.concat([df, feedback_df])


# Prepare features and labels
X = df["text"]
y = df["label"]
groups = df["group"]

# load the encoder and encode everything (important for embeddings vs simple TF)
encoder = SentenceTransformer("all-MiniLM-L6-v2")
X_encoded = encoder.encode(X.tolist(), show_progress_bar=True)

# Train/test split — GroupShuffleSplit (instead of a plain random split) keeps every
# sentence from the same source "No Distortion" paragraph on one side of the split,
# so the test set is never evaluated on sentences whose vocabulary/tone the model
# already saw in train via a sibling sentence from the same paragraph.
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X_encoded, y, groups=groups))

X_train, X_test = X_encoded[train_idx], X_encoded[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Train a Logistic Regression classifier on top of the sentence embeddings.
# Instead of TF-IDF word frequency vectors, LogReg now receives 384-dimensional
# semantic embeddings from the SentenceTransformer encoder — this means sentences
# with similar meanings will have similar input vectors, giving LogReg much richer
# signal to learn from than raw word counts.
# class_weight='balanced' handles class imbalance — without it, the model would
# over-predict common classes (like "No Distortion") and ignore rare ones.
# max_iter=1000 ensures the solver has enough iterations to converge on optimal weights.
# C=1.0 is the regularization strength — it penalizes overly large weights so the
# model doesn't over-fit to quirks in the training data.
# lbfgs is the solver algorithm used to find those optimal weights — well suited for
# multiclass problems with dense input like our 384-dimensional embeddings.
pipeline = LogisticRegression(
      max_iter=1000,
      class_weight="balanced",
      C=1.0,
      solver="lbfgs",
  )
    
    


# fit the model to the training data
pipeline.fit(X_train, y_train)

# run the prediction method from the pipeline/LR model on the test data
y_pred = pipeline.predict(X_test)


# using the accuracy method, compute the overall accuracy of the model on the test data
accuracy = accuracy_score(y_test, y_pred)
print(f"\nOverall Accuracy: {accuracy:.2%}")

# print out some metrics about the training and test sets
print(f"\nTraining set size: {len(y_train)}")
print(f"Test set size: {len(y_test)}")
print(f"\nClass distribution in test set:")
print(y_test.value_counts().sort_index())

# using the classification report method from sklearn, print out a detailed classification report, which
# contains info such as precision, recall, and F1-score for each class
print("Classification Report:")

print(classification_report(y_test, y_pred))

# save the model to a file and essentially cache it for later use in the Flask API
# this reduces latency for the end user since we don't have to retrain the model on every request / text that the user enters
joblib.dump(pipeline, "distortion_model.pkl")

try:
    if len(feedback_data) > 0:
        mark_used_feedback()
except Exception as e:
    print(f"Error with marking feedback as used in the database: {e}")

try:
    new_version = get_latest_version() + 1
    save_model_version(
        new_version,
        len(X_train),
        accuracy,
        f"Trained with {len(feedback_data)} new feedback samples",
    )
except Exception as e:
    print(f"Error with saving model: {e}")

print("Model successfully saved!")
