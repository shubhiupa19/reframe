from sklearn.model_selection import GridSearchCV
import pandas as pd
import re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer


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
has_distortion_clean = has_distortion[["Distorted part", "Dominant Distortion"]]
has_distortion_clean.columns = ["text", "label"]

# for no distortion, split each paragraph into sentences
no_distortion_sentences = []
for _, row in no_distortion.iterrows():
    sentences = re.split(r'(?<=[.!?])\s+', row["Patient Question"].strip())
    for sentence in sentences:
        if sentence.strip():
            no_distortion_sentences.append({"text": sentence.strip(), "label": "No Distortion"})

no_distortion_clean = pd.DataFrame(no_distortion_sentences)
# Downsample "No Distortion" to ~400 to mitigate class imbalance
no_distortion_downsampled = no_distortion_clean.sample(n=400, random_state=42)

# Store this downsample + rest of data in the primary df
kaggle_clean_df = pd.concat([has_distortion_clean, no_distortion_downsampled])

# combine with augmented data
df = pd.concat([kaggle_clean_df, augmented_df])

# Prepare features and labels
X = df["text"]
y = df["label"]

# load the encoder and encode everything (important for embeddings vs simple TF)
# this must match train_model.py's approach so tuning results transfer to the real model
encoder = SentenceTransformer("all-MiniLM-L6-v2")
X_encoded = encoder.encode(X.tolist(), show_progress_bar=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

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

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro', verbose=2)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best F1 score: {grid_search.best_score_:.2%}")