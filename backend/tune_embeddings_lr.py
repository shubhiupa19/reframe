from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

from data_prep import build_training_dataset

df, _ = build_training_dataset(include_feedback=False)

# Prepare features and labels
X = df["text"]
y = df["label"]
groups = df["group"]

# load the encoder and encode everything (important for embeddings vs simple TF)
# this must match train_embeddings_lr.py's approach so tuning results transfer to the real model
encoder = SentenceTransformer("all-MiniLM-L6-v2")
X_encoded = encoder.encode(X.tolist(), show_progress_bar=True)

# Train/test split — GroupShuffleSplit (instead of a plain random split) keeps every
# sentence from the same source "No Distortion" paragraph on one side, matching the
# leakage fix used in train_embeddings_lr.py.
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X_encoded, y, groups=groups))

X_train, X_test = X_encoded[train_idx], X_encoded[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train = groups.iloc[train_idx]

# Pipeline: scale the raw embeddings first, then feed them to Logistic Regression.
# SentenceTransformer embeddings aren't normalized/scaled by default, and LogReg's
# regularization (the C param) assumes features are on comparable scales — StandardScaler
# (mean 0, unit variance per dimension) fixes that so the C penalty is applied fairly
# across all 384 embedding dimensions instead of being skewed by dimensions with larger magnitudes.
# we are also using class_weight='balanced' to handle any class imbalance, which occurs when classes (types of distortions) are not equally represented in the dataset
# this imbalance is problematic because the model may become biased towards the majority class (No Distortion in this case) and perform poorly on minority classes
# we are also setting max_iter to 1000 to ensure convergence, which means the model has enough iterations to find the optimal solution
# solver and penalty are left out of the constructor now since we're grid searching both below
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
            ),
        ),
    ]
)

# C controls regularization strength, which reduces weights on features that aren't
# indicative of a particular distortion type.
# penalty controls which kind of regularization is applied: l2 shrinks all weights
# smoothly, while l1 can zero out weights entirely (implicit feature selection).
# solver is the optimization algorithm used to fit the weights — lbfgs only supports
# the l2 penalty, while saga supports both l1 and l2, so we group them into separate
# grids (a list of dicts) to avoid GridSearchCV trying invalid solver/penalty combos.
param_grid = [
    {
        'clf__solver': ['lbfgs'],
        'clf__penalty': ['l2'],
        'clf__C': [0.01, 0.1, 1.0, 10.0],
    },
    {
        'clf__solver': ['saga'],
        'clf__penalty': ['l1', 'l2'],
        'clf__C': [0.01, 0.1, 1.0, 10.0],
    },
]

# GroupKFold (instead of the default integer cv, which is a plain KFold) keeps every
# sentence from the same source paragraph together within each cross-validation fold,
# not just in the outer train/test split above — otherwise a fold could still evaluate
# on "No Distortion" sentences whose sibling from the same paragraph was in that fold's
# training portion.
grid_search = GridSearchCV(pipeline, param_grid, cv=GroupKFold(n_splits=5), scoring='f1_macro', verbose=2)
grid_search.fit(X_train, y_train, groups=groups_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best F1 score: {grid_search.best_score_:.2%}")
