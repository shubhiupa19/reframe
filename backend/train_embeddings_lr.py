from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, accuracy_score
import joblib

from data_prep import build_training_dataset
from database import (
    mark_used_feedback,
    save_model_version,
    get_latest_version,
)

df, feedback_data = build_training_dataset(include_feedback=True)

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
